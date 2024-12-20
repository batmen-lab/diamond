import argparse
import itertools
import json
import logging
import os

import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb
import xlearn as xl
from kan import KAN
from lightning.pytorch import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from path_explain import PathExplainerTorch
from shap import TreeExplainer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAUROC

from knockoffs import (gen_DeepKnockoffs, gen_KnockoffGAN, gen_Knockoffsdiag,
                       gen_VAEKnockoff)
from models import DeepPink
from simulation import gen_X, gen_Y


class DataModule(LightningDataModule):
    def __init__(self, X_train, X_val, X_test, Y_train, Y_val, Y_test, batch_size=256):
        super().__init__()
        self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
        self.test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
    

class DeepPinkModel(LightningModule):
    def __init__(self, model, criterion, optimizer_class=Adam, optimizer_params=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params or {"lr": 1e-3}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Optionally log metrics (e.g., AUC for binary classification)
        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            preds = torch.sigmoid(y_hat)
            metric = BinaryAUROC().to(preds.device)
            auc = metric(preds, y)
            self.log('val_auc', auc, on_epoch=True, prog_bar=True)


def generate_and_preprocess_data(num_samples, num_features, func_num, knockoff, seed):
    # Generate data
    X = gen_X.UniformSampler(p=num_features, low_list=([0]*num_features), high_list=([1]*num_features), seed=seed).sample(n=num_samples)
    Y, inter_gt, import_gt = gen_Y.generate_interaction_response(X, func_num=func_num)
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # Generating knockoff X
    if knockoff == 'knockoffgan':
        knockoff_func, release_memory = gen_KnockoffGAN.KnockoffGAN(X_scaled, "Uniform", seed=seed)
        X_knockoff = knockoff_func(X_scaled)
        release_memory()
    elif knockoff == 'deepknockoffs':
        X_knockoff = gen_DeepKnockoffs.DeepKnockoffs(X_scaled, batch_size=512, test_size=0, nepochs=10, epoch_length=100)
    elif knockoff == 'vaeknockoff':
        X_knockoff = gen_VAEKnockoff.train_vae_knockoff(X_scaled, n_epoch=100, mb_size=256)
    elif knockoff == 'knockoffsdiag':
        X_knockoff = gen_Knockoffsdiag.conditional_sequential_gen_ko(X_scaled, np.zeros(X_scaled.shape[1]), n_jobs=20, seed=seed)
    elif knockoff == 'permutation':
        X_knockoff = np.random.permutation(X_scaled.T).T
    elif knockoff == 'self':
        X_knockoff = X_scaled
    X_concat = np.concatenate((X_scaled, X_knockoff), axis=1)

    # Split data into training, validation, and test sets
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X_concat, Y_scaled, test_size=0.3, random_state=seed)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=seed)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, inter_gt, import_gt

def train_and_explain(seed, func_num, model_type, knockoff, *args, **kwargs):
    # Create output directory if it does not exist
    output_dir = f'../output/simulation/{model_type}_{knockoff}'
    if model_type in ['mlp', 'cnn', 'transformer', 'kan']:
        output_dir += f'_{kwargs["explainer"]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up logging to suppress unnecessary output
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

    # Set up seed for reproducibility
    seed_everything(seed)

    # Generate and preprocess data
    num_features = 30
    num_samples = 20000
    X_train, X_val, X_test, Y_train, Y_val, Y_test, inter_gt, import_gt = generate_and_preprocess_data(num_samples, num_features, func_num, knockoff, seed)

    if model_type in ['mlp', 'cnn', 'transformer']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_module = DataModule(X_train, X_val, X_test, Y_train, Y_val, Y_test)

        # Train the model
        if model_type == 'mlp':
            model = DeepPink.DeepPINK(p=num_features, 
                                    model_type='mlp', 
                                    hidden_dims=[140, 100, 60, 20])
        elif model_type == 'cnn':
            model = DeepPink.DeepPINK(p=num_features, model_type='cnn')
        elif model_type == 'transformer':
            model = DeepPink.DeepPINK(p=num_features, model_type='transformer') 
        model = DeepPinkModel(model, criterion=torch.nn.MSELoss())
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

        trainer = Trainer(max_epochs=100, 
                        accelerator='gpu',
                        callbacks=[early_stop_callback, checkpoint_callback],
                        enable_checkpointing=True)
        trainer.fit(model, data_module)

        # Load the best model
        model = DeepPinkModel.load_from_checkpoint(checkpoint_callback.best_model_path, model=model.model, criterion=model.criterion)

        if kwargs['explainer'] == 'topo' and model_type == 'mlp':
            attributions, interactions = model.model.global_feature_interactions()
        else:
            # Get the test dataset
            test_dataset = torch.tensor(X_test, dtype=torch.float32, device=device)
            test_dataset.requires_grad = True

            if kwargs['explainer'] == 'ig':
                baseline = torch.zeros((1, num_features * 2), device=device)
                use_expectation = False
            elif kwargs['explainer'] == 'eg':
                baseline = torch.tensor(X_train, dtype=torch.float32, device=device)
                use_expectation = True
            else:
                RuntimeError('Invalid explainer')

            # Initialize the explainer with the model
            model.model.to(device)
            explainer = PathExplainerTorch(model.model)

            attributions = explainer.attributions(test_dataset, baseline=baseline, use_expectation=use_expectation).detach().cpu().numpy()
            interactions = explainer.interactions(test_dataset, baseline=baseline, use_expectation=use_expectation).detach().cpu().numpy()

    elif model_type == 'kan':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = {"train_input": torch.tensor(X_train, dtype=torch.float32, device=device),
                   "train_label": torch.tensor(Y_train, dtype=torch.float32, device=device),
                   "test_input": torch.tensor(X_val, dtype=torch.float32, device=device),
                   "test_label": torch.tensor(Y_val, dtype=torch.float32, device=device)}

        # Train the model
        model = KAN(width=[2 * num_features, num_features // 2, 1], device=device, seed=seed, auto_save=False)
        model.fit(dataset, steps=100)

        # Clear the cache
        torch.cuda.empty_cache()

        # Get the test dataset
        test_dataset = torch.tensor(X_test, dtype=torch.float32, device=device)
        test_dataset.requires_grad = True

        if kwargs['explainer'] == 'ig':
            baseline = torch.zeros((1, num_features * 2), device=device)
            use_expectation = False
        elif kwargs['explainer'] == 'eg':
            baseline = torch.tensor(X_train, dtype=torch.float32, device=device)
            use_expectation = True
        else:
            RuntimeError('Invalid explainer')

        # Initialize the explainer with the model
        explainer = PathExplainerTorch(model)

        # Compute attributions and interactions
        attributions = explainer.attributions(test_dataset, baseline=baseline, use_expectation=use_expectation).detach().cpu().numpy()
        interactions = explainer.interactions(test_dataset, baseline=baseline, use_expectation=use_expectation).detach().cpu().numpy()

    elif model_type == 'xgboost':
        # Train the model
        model = xgb.XGBRegressor(callbacks=[xgb.callback.EarlyStopping(rounds=10, metric_name='rmse', save_best=True)],
                                 device='cuda' if torch.cuda.is_available() else 'cpu',
                                 random_state=seed)
        model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)])

        # Initialize the explainer with the model
        explainer = TreeExplainer(model)

        # Compute attributions and interactions
        attributions = explainer.shap_values(X=X_test)
        interactions = explainer.shap_interaction_values(X=X_test)

    elif model_type == 'lightgbm':
        # Train the model
        model = lgb.LGBMRegressor(device='gpu' if torch.cuda.is_available() else 'cpu',
                                  random_state=seed)
        model.fit(X_train, Y_train.ravel(), 
                  eval_set=[(X_val, Y_val.ravel())], 
                  callbacks=[lgb.early_stopping(stopping_rounds=10)])

        # Initialize the explainer with the model
        explainer = TreeExplainer(model)

        # Compute attributions and interactions
        attributions = explainer.shap_values(X=X_test)
        interactions = explainer.shap_interaction_values(X=X_test)
        
    elif model_type == 'fm':
        # Convert data to DMatrix format
        dtrain = xl.DMatrix(X_train, label=Y_train.ravel())
        dval = xl.DMatrix(X_val, label=Y_val.ravel())

        # Train the model
        model = xl.create_fm()
        model.setTrain(dtrain)
        model.setValidate(dval)
        param = {"task": "reg", "metric": "rmse", "epoch": 100, "stop_window": 10}
        model.setTXTModel(f"{output_dir}/func{func_num}_seed{seed}.txt")
        model.fit(param, f"{output_dir}/func{func_num}_seed{seed}.out")

        # Compute attributions and interactions
        ws = []
        vs = []
        with open(f"{output_dir}/func{func_num}_seed{seed}.txt") as f:
            for line in f:
                if line.startswith('i'):
                    ws.append(float(line.split()[1]))
                elif line.startswith('v'):
                    vs.append(np.array([float(x) for x in line.split()[1:]]))

        attributions = np.zeros((1, num_features * 2))
        interactions = np.zeros((1, num_features * 2, num_features * 2))
        for i in range(num_features * 2):
            attributions[0, i] = ws[i]
            interactions[0, i, i] = np.dot(vs[i], vs[i])
        for i, j in itertools.combinations(range(num_features * 2), 2):
            interactions[0, i, j] = np.dot(vs[i], vs[j])
            interactions[0, j, i] = interactions[0, i, j]

        # Clean up the files
        os.remove(f"{output_dir}/func{func_num}_seed{seed}.out")
        os.remove(f"{output_dir}/func{func_num}_seed{seed}.txt")
                
    else:
        raise ValueError('Invalid model type')

    # Get the indices of the true features, true pairs, false pairs, original knockoff pairs, and knockoff knockoff pairs
    true_indices = sorted(import_gt)
    true_pairs = list(set([(i, j) for features in inter_gt for i, j in itertools.combinations(sorted(list(features)), 2)]))
    false_pairs = [(i, j) for i, j in itertools.product(range(num_features), range(num_features)) if i < j and (i, j) not in true_pairs]
    original_knockoff_pairs = [(i, j) for i, j in itertools.product(range(num_features), range(num_features, 2*num_features)) if i < j and i != j - num_features]
    knockoff_knockoff_pairs = [(i, j) for i, j in itertools.product(range(num_features, 2*num_features), range(num_features, 2*num_features)) if i < j]

    # Sum the attributions and interactions over the samples
    attributions = np.mean(np.abs(attributions), axis=0)
    interactions = np.mean(np.abs(interactions), axis=0)

    # Save the results to a JSON file
    results_to_save = {
        'seed': seed,
        'func_num': func_num,
        'true_indices': true_indices,
        'true_pairs': true_pairs,
        'false_pairs': false_pairs,
        'original_knockoff_pairs': original_knockoff_pairs,
        'knockoff_knockoff_pairs': knockoff_knockoff_pairs,
        'interactions': interactions.tolist(),
        'attributions': attributions.tolist(),
    }
    with open(f'{output_dir}/func{func_num}_seed{seed}.json', 'w') as f:
        json.dump(results_to_save, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--func_num', type=int, required=True)
    parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn', 'transformer', 'kan', 'xgboost', 'lightgbm', 'rf', 'fm'], required=True)
    parser.add_argument('--knockoff', type=str, default='knockoffsdiag', choices=["knockoffgan", "deepknockoffs", 'vaeknockoff', 'knockoffsdiag', 'permutation', 'self'])
    parser.add_argument('--explainer', type=str, default='eg', choices=['ig', 'eg', 'topo'])
    args = parser.parse_args()

    train_and_explain(args.seed, args.func_num, args.model_type, args.knockoff, explainer=args.explainer)
