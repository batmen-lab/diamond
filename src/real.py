import argparse
import itertools
import json
import logging
import os

import lightgbm as lgb
import numpy as np
import shap
import torch
import xgboost as xgb
import xlearn as xl
from lightning.pytorch import Trainer, seed_everything
from path_explain import PathExplainerTorch
from sklearn.utils import compute_class_weight

from knockoffs import (gen_DeepKnockoffs, gen_KnockoffGAN, gen_Knockoffsdiag,
                       gen_VAEKnockoff)
from models import model_utils
from sim import DataModule, DeepPINKModel
from utils import prep_data


def load_and_preprocess_data(dataset, knockoff, seed):
    if dataset == "enhancer":
        data_url = "../data/enhancer/enhancer.Rdata"
        X, Y, feat_names = prep_data.load_enhancer_data(data_url, group_func="mean")
        name_import_gt, named_inter_gt = prep_data.get_drosophila_enhancer_gt()
        inter_gt = prep_data.inter_gt_name2idx(named_inter_gt, feat_names)
        import_gt = prep_data.import_gt_name2idx(name_import_gt, feat_names)
        task = 'classification'
    elif dataset == "mortality":
        data_dir = "../data/mortality"
        X, Y, feat_names = prep_data.load_mortality_data(data_dir)
        name_import_gt, named_inter_gt = prep_data.get_mortality_gt()
        inter_gt = prep_data.inter_gt_name2idx(named_inter_gt, feat_names)
        import_gt = prep_data.import_gt_name2idx(name_import_gt, feat_names)
        task = 'regression'
    elif dataset == "diabetes":
        X, Y, feat_names = prep_data.load_diabetes_data()
        inter_gt, import_gt = [], []
        task = 'regression'
    elif dataset == "cal_housing":
        X, Y, feat_names = prep_data.load_cal_housing_data()
        inter_gt, import_gt = [], []
        task = 'regression'
    else:
        raise ValueError("Invalid dataset: {}".format(dataset))
    
    # Generating knockoff X
    if knockoff == 'knockoffgan':
        knockoff_func, release_memory = gen_KnockoffGAN.KnockoffGAN(X, "Uniform", seed=seed)
        X_knockoff = knockoff_func(X)
        release_memory()
    elif knockoff == 'deepknockoffs':
        X_knockoff = gen_DeepKnockoffs.DeepKnockoffs(X, batch_size=512, test_size=0, nepochs=10, epoch_length=100)
    elif knockoff == 'vaeknockoff':
        X_knockoff = gen_VAEKnockoff.train_vae_knockoff(X, n_epoch=100, mb_size=256)
    elif knockoff == 'knockoffsdiag':
        X_knockoff = gen_Knockoffsdiag.conditional_sequential_gen_ko(X, np.zeros(X.shape[1]), n_jobs=40, seed=seed)
    X_concat = np.concatenate((X, X_knockoff), axis=1)

    return X_concat, Y, inter_gt, import_gt, feat_names, task

def train_and_explain(seed, dataset, model_type, knockoff, *args, **kwargs):
    # Create output directory if it does not exist
    output_dir = f'../output/real/{dataset}/{model_type}_{knockoff}'
    if model_type == 'nn':
        output_dir += f'_{kwargs["explainer"]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up logging to suppress unnecessary output
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

    # Set up seed for reproducibility
    seed_everything(seed)

    # Load and preprocess the data
    X, Y, inter_gt, import_gt, feat_names, task = load_and_preprocess_data(dataset, knockoff, seed)
    num_features = X.shape[1] // 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Define the criterion
    if task == 'classification':
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y.ravel())
        pos_weight = class_weights[1] / class_weights[0]

    if model_type == 'nn':
        data_module = DataModule(X, X, X, Y, Y, Y, batch_size=128)

        # Define the criterion
        if task == 'classification':
            pos_weight = torch.tensor([pos_weight], device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if dataset == "mortality":
                criterion = model_utils.CoxPHLoss()
            else:
                criterion = torch.nn.MSELoss()

        # Train the model
        model = DeepPINKModel(num_features=num_features, 
                              pairwise_layer=True, 
                              criterion=criterion)
        trainer = Trainer(max_epochs=100, 
                          accelerator='gpu' if torch.cuda.is_available() else 'cpu')
        trainer.fit(model, data_module)

        if kwargs['explainer'] == 'topo':
            attributions, interactions = model.model.global_feature_interactions()
        else:
            # Get the test dataset
            test_dataset = torch.tensor(X, dtype=torch.float32, device=model.device)
            test_dataset.requires_grad = True

            if kwargs['explainer'] == 'ig':
                baseline = torch.zeros((1, num_features * 2), device=model.device)
                use_expectation = False
            elif kwargs['explainer'] == 'eg':
                baseline = torch.tensor(X, dtype=torch.float32, device=model.device)
                use_expectation = True
            else:
                RuntimeError('Invalid explainer')

            # Initialize the explainer with the model
            explainer = PathExplainerTorch(model.model)

            # Compute attributions and interactions
            attributions = explainer.attributions(test_dataset, baseline=baseline, use_expectation=use_expectation).detach().cpu().numpy()
            interactions = explainer.interactions(test_dataset, baseline=baseline, use_expectation=use_expectation).detach().cpu().numpy()

    elif model_type == 'xgboost':
        # Train the model
        if task == 'regression':
            model = xgb.XGBRegressor(n_estimators=100,
                                    eval_metric='rmse',
                                    device=device,
                                    random_state=seed)
        else:
            model = xgb.XGBClassifier(n_estimators=100,
                                    scale_pos_weight=pos_weight,
                                    device=device,
                                    random_state=seed)
        model.fit(X, Y)

        # Initialize the explainer with the model
        explainer = shap.TreeExplainer(model)

        # Compute attributions and interactions
        attributions = explainer.shap_values(X=X)
        interactions = explainer.shap_interaction_values(X=X)

    elif model_type == 'lightgbm':
        # Train the model
        if task == 'regression':
            model = lgb.LGBMRegressor(n_estimators=100,
                                   device='gpu' if torch.cuda.is_available() else 'cpu',
                                   random_state=seed)
        else:
            model = lgb.LGBMClassifier(n_estimators=100,
                                    scale_pos_weight=pos_weight, 
                                    device='gpu' if torch.cuda.is_available() else 'cpu',
                                    random_state=seed)
        model.fit(X, Y.ravel())

        # Initialize the explainer with the model
        explainer = shap.TreeExplainer(model)

        # Compute attributions and interactions
        attributions = explainer.shap_values(X=X)
        interactions = explainer.shap_interaction_values(X=X)
        
    elif model_type == 'fm':
        # Convert data to DMatrix format
        dtrain = xl.DMatrix(X, label=Y.ravel())

        # Train the model
        model = xl.create_fm()
        model.setTrain(dtrain)
        if task == 'regression':
            param = {"task": "reg", "metric": "rmse", "epoch": 100}
        else:
            param = {"task": "binary", "metric": "auc", "epoch": 100}
        model.setTXTModel(f"{output_dir}/{dataset}_seed{seed}.txt")
        model.fit(param, f"{output_dir}/{dataset}_seed{seed}.out")

        # Compute attributions and interactions
        ws = []
        vs = []
        with open(f"{output_dir}/{dataset}_seed{seed}.txt") as f:
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
        os.remove(f"{output_dir}/{dataset}_seed{seed}.out")
        os.remove(f"{output_dir}/{dataset}_seed{seed}.txt")
                
    else:
        raise ValueError(f"Invalid model type")
    
    # Get the indices of the true features, true pairs, false pairs, original knockoff pairs, and knockoff knockoff pairs
    true_indices = sorted(import_gt)
    true_pairs = [(i, j) for features in inter_gt for i, j in itertools.combinations(sorted(list(features)), 2)]
    false_pairs = [(i, j) for i, j in itertools.product(range(num_features), range(num_features)) if i < j and (i, j) not in true_pairs]
    original_knockoff_pairs = [(i, j) for i, j in itertools.product(range(num_features), range(num_features, 2*num_features)) if i < j and i != j - num_features]
    knockoff_knockoff_pairs = [(i, j) for i, j in itertools.product(range(num_features, 2*num_features), range(num_features, 2*num_features)) if i < j]

    # Average the attributions and interactions across samples
    attributions = np.array(attributions)
    interactions = np.array(interactions)
    
    if attributions.ndim == 3:
        indices = Y.astype(int).ravel()
        attributions = attributions[indices, np.arange(len(indices)), :]
        attributions = attributions.reshape((-1, num_features * 2))

    if 'save_local' in kwargs and kwargs['save_local']:
        if seed == 1 and model_type != 'fm' and ('explainer' not in kwargs or kwargs['explainer'] != 'topo'):
            np.save(f'{output_dir}/{dataset}_seed{seed}_local_attributions.npy', attributions)
            np.save(f'{output_dir}/{dataset}_seed{seed}_local_interactions.npy', interactions)
            np.save(f'{output_dir}/{dataset}_seed{seed}_X.npy', X)
     
    attributions = np.mean(np.abs(attributions), axis=0)
    interactions = np.mean(np.abs(interactions), axis=0)

    # Save the results to a JSON file
    results_to_save = {
        'seed': seed,
        'dataset': dataset,
        'feat_names': feat_names.tolist() if isinstance(feat_names, np.ndarray) else feat_names,
        'true_indices': true_indices,
        'true_pairs': true_pairs,
        'false_pairs': false_pairs,
        'original_knockoff_pairs': original_knockoff_pairs,
        'knockoff_knockoff_pairs': knockoff_knockoff_pairs,
        'interactions': interactions.tolist(),
        'attributions': attributions.tolist()
    }           

    with open(f'{output_dir}/{dataset}_seed{seed}.json', 'w') as f:
        json.dump(results_to_save, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, choices=['enhancer', 'mortality', 'diabetes', 'breast_cancer', 'cal_housing', 'bike_sharing'], required=True)
    parser.add_argument('--model_type', type=str, choices=['nn', 'xgboost', 'lightgbm', 'fm'], default='nn')
    parser.add_argument('--knockoff', type=str, default='gan', choices=["knockoffgan", "deepknockoffs", 'vaeknockoff', 'knockoffsdiag'])
    parser.add_argument('--explainer', type=str, choices=['ig', 'eg', 'topo'], default='ig')
    parser.add_argument('--save_local', action='store_true')
    args = parser.parse_args()

    train_and_explain(args.seed, args.dataset, args.model_type, args.knockoff, explainer=args.explainer, save_local=args.save_local)


