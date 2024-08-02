import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from pygam import LinearGAM
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from models import model_utils


def prepare_attribution_dataframe(data: dict) -> pd.DataFrame:
    attributions = np.array(data['attributions'])
    feat_names = data['feat_names'] + [f'knockoff_{name}' for name in data['feat_names']] if 'feat_names' in data else [f'feature_{i}' for i in range(len(attributions // 2))] + [f'knockoff_{i}' for i in range(len(attributions // 2))]
    attribution_records = []
    for i, attr in enumerate(attributions):
        record = {
            'seed': data['seed'],
            'dataset': f'F{data["func_num"]}' if 'func_num' in data else data['dataset'],
            'feature': i,
            'name': feat_names[i],
            'attribution': attr,
            'type': 'Original' if i < len(attributions) // 2 else 'Knockoff',
            'subtype': 'True' if i in data['true_indices'] else 'False',
        }
        attribution_records.append(record)
    
    attribution_df = pd.DataFrame(attribution_records)

    return attribution_df

def prepare_interaction_dataframe(data: dict) -> pd.DataFrame:
    attributions = np.array(data['attributions'])
    interactions = np.array(data['interactions'])

    index_groups = {key: [(i, j) for i, j in data[key]] for key in ['true_pairs', 'false_pairs', 'original_knockoff_pairs', 'knockoff_knockoff_pairs']}
    original_pairs = index_groups['true_pairs'] + index_groups['false_pairs']
    knockoff_pairs = index_groups['original_knockoff_pairs'] + index_groups['knockoff_knockoff_pairs']
    feat_names = data['feat_names'] + [f'knockoff_{name}' for name in data['feat_names']] if 'feat_names' in data else [f'feature_{i}' for i in range(len(attributions // 2))] + [f'knockoff_{i}' for i in range(len(attributions // 2))]

    interaction_records = []
    for (i, j), subtype in zip(original_pairs + knockoff_pairs, 
                               ['True']*len(index_groups['true_pairs']) + 
                               ['False']*len(index_groups['false_pairs']) + 
                               ['Original-Knockoff']*len(index_groups['original_knockoff_pairs']) + 
                               ['Knockoff-Knockoff']*len(index_groups['knockoff_knockoff_pairs'])):
        record = {
            'seed': data['seed'],
            'dataset': f'F{data["func_num"]}' if 'func_num' in data else data['dataset'],
            'pair': (i, j),
            'name': f'{feat_names[i]} - {feat_names[j]}',
            'type': 'Original' if (i, j) in original_pairs else 'Knockoff',
            'subtype': subtype,
            'interaction': interactions[i, j],
            'feature1': i,
            'feature2': j,
            'pair_onehot': np.eye(len(attributions))[i] + np.eye(len(attributions))[j],
            'attribution1': attributions[i],
            'attribution2': attributions[j],
            'attr_onehot': np.eye(len(attributions))[i] * attributions[i] + np.eye(len(attributions))[j] * attributions[j]
        }

        interaction_records.append(record)

    interaction_df = pd.DataFrame(interaction_records)

    return interaction_df


def fdr_control(data_dir: str, target_fdr: float = 0.2) -> None:
    # Set output directory and file paths
    output_dir = f'{data_dir}/fdr_control'
    os.makedirs(output_dir, exist_ok=True)
    attributions_df_path = f'{output_dir}/attributions.csv'
    interactions_df_path = f'{output_dir}/interactions.csv'
    fdr_power_df_path = f'{output_dir}/fdr_power.csv'
    
    if all([os.path.exists(f) for f in [attributions_df_path, interactions_df_path, fdr_power_df_path]]):
        print('FDR control results already exist')
    else:
        print('Running FDR control')
        result_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        attributions_dfs = []
        interactions_dfs = []
        fdr_power_records = []

        # Process data by dataset
        dataset_files = {}
        for result_file in result_files:
            with open(f'{data_dir}/{result_file}', 'r') as file:
                try:
                    data = json.load(file)
                    dataset = f'F{data["func_num"]}' if 'func_num' in data else data['dataset']
                    if dataset not in dataset_files:
                        dataset_files[dataset] = []
                    dataset_files[dataset].append(result_file)
                except json.JSONDecodeError:
                    continue

        simulation = True if 'func_num' in data else False

        for dataset, files in dataset_files.items():
            num_rounds = min(20, len(files)) if simulation else 20
            for round_idx in tqdm(range(num_rounds)):
                # Set seed
                random.seed(round_idx)
                np.random.seed(round_idx)

                # Randomly select 5 files for each round if not simulation else select 1 file
                round_attributions_dfs = []
                round_interactions_dfs = []
                selected_files = random.sample(files, min(5, len(files))) if not simulation else files[round_idx:round_idx+1]

                for result_file in selected_files:
                    # Load data
                    with open(f'{data_dir}/{result_file}', 'r') as file:
                        try:
                            data = json.load(file)
                        except json.JSONDecodeError:
                            continue

                    # Parse data from JSON
                    num_features = np.array(data['attributions']).shape[-1]
                    interactions_df = prepare_interaction_dataframe(data)
                    attributions_df = prepare_attribution_dataframe(data)
                    round_attributions_dfs.append(attributions_df)
                    round_interactions_dfs.append(interactions_df)

                if len(round_attributions_dfs) > 1:
                    round_attributions_df = pd.concat(round_attributions_dfs)
                    round_interactions_df = pd.concat(round_interactions_dfs)
                else:
                    round_attributions_df = pd.DataFrame(round_attributions_dfs[0])
                    round_interactions_df = pd.DataFrame(round_interactions_dfs[0])
                round_attributions_df = round_attributions_df.groupby(['dataset', 'feature']).agg({
                    'name': 'first',
                    'type': 'first',
                    'subtype': 'first',
                    'attribution': 'mean'
                }).reset_index()
                round_interactions_df = round_interactions_df.groupby(['dataset', 'pair']).agg({
                    'name': 'first',
                    'type': 'first',
                    'subtype': 'first',
                    'interaction': 'mean',
                    'feature1': 'first',
                    'feature2': 'first',
                    'pair_onehot': 'first',
                    'attribution1': 'mean',
                    'attribution2': 'mean',
                    'attr_onehot': 'mean'
                }).reset_index()

                round_attributions_df['seed'] = round_idx
                round_interactions_df['seed'] = round_idx

                # Prepare data for calibration
                round_interactions_df['calibrated_interaction'] = round_interactions_df['interaction']
                attr_onehot = np.array(round_interactions_df['attr_onehot'].values.tolist())
                pair_onehot = np.array(round_interactions_df['pair_onehot'].values.tolist())
                if 'xgboost' in data_dir or 'lightgbm' in data_dir or 'fm' in data_dir:
                    projection_matrix = np.random.normal(size=(num_features, num_features))
                else:
                    projection_matrix = np.random.normal(size=(num_features, int(1/3 * num_features)))
                projected_pair_onehot = pair_onehot @ projection_matrix

                # Calculate propensity scores
                X_train = attr_onehot
                y_train = round_interactions_df['type'].apply(lambda x: 1 if x == 'Original' else 0).values
                model = LogisticRegression().fit(X_train, y_train)
                propensity_scores = model.predict_proba(X_train)[:, 1]
                round_interactions_df['propensity_score'] = propensity_scores

                # Calculate stabilized IPTW
                stabilized_factor = np.mean(y_train)
                round_interactions_df['IPTW'] = round_interactions_df.apply(lambda row: stabilized_factor / row['propensity_score'] if row['type'] == 'Original' else (1 - stabilized_factor) / (1 - row['propensity_score']), axis=1)

                X_test = np.concatenate([attr_onehot, projected_pair_onehot], axis=1)
                obs = round_interactions_df['calibrated_interaction'].values

                X_train = X_test.copy()
                y_train = obs.copy()

                # Train model to calibrate interactions
                model = LinearGAM(constraints=None, penalties=None)
                model.fit(X_train, y_train, weights=1 / round_interactions_df['IPTW'].values)

                pred = model.predict(X_test)
                residual = np.abs(obs - pred)

                round_interactions_df['obs'] = obs
                round_interactions_df['pred'] = pred
                round_interactions_df['calibrated_interaction'] = residual

                record = {
                    'seed': round_idx,
                    'dataset': dataset,
                }

                uncalibrated_interactions = [(row['pair'], row['interaction']) for _, row in round_interactions_df.iterrows()]
                calibrated_interactions = [(row['pair'], row['calibrated_interaction']) for _, row in round_interactions_df.iterrows()]
                uncalibrated_interactions_original = [(row['pair'], row['interaction']) for _, row in round_interactions_df.iterrows() if row['type'] == 'Original']
                calibrated_interactions_original = [(row['pair'], row['calibrated_interaction']) for _, row in round_interactions_df.iterrows() if row['type'] == 'Original']

                # Get selected interactions
                selected_uncalibrated, threshold_uncalibrated = model_utils.get_selected_interactions(uncalibrated_interactions, num_features // 2, target_fdr)
                selected_calibrated, threshold_calibrated = model_utils.get_selected_interactions(calibrated_interactions, num_features // 2, target_fdr)
                true_pairs = [set(pair) for pair in round_interactions_df[round_interactions_df['subtype'] == 'True']['pair']]
                false_pairs = [pair for pair in round_interactions_df[round_interactions_df['subtype'] == 'False']['pair']]

                # Calculate metrics
                if len(true_pairs) > 0:
                    uncalibrated_auc = model_utils.pairwise_interaction_auc(uncalibrated_interactions_original, true_pairs)
                    calibrated_auc = model_utils.pairwise_interaction_auc(calibrated_interactions_original, true_pairs)
                else:
                    uncalibrated_auc = 0
                    calibrated_auc = 0

                uncalibrated_fdr = model_utils.get_interaction_fdp(selected_uncalibrated, true_pairs)
                calibrated_fdr = model_utils.get_interaction_fdp(selected_calibrated, true_pairs)
            
                uncalibrated_power = model_utils.get_interaction_power(selected_uncalibrated, true_pairs)
                calibrated_power = model_utils.get_interaction_power(selected_calibrated, true_pairs)

                uncalibrated_q_values = model_utils.get_q_values(uncalibrated_interactions, num_features // 2)
                calibrated_q_values = model_utils.get_q_values(calibrated_interactions, num_features // 2)

                '''
                if calibrated_fdr > target_fdr:
                    print(len(selected_calibrated))
                    print(set([pair for pair, _ in selected_calibrated]).intersection(false_pairs))
                    print('Dataset:', dataset)
                    print(f'Uncalibrated FDR:', uncalibrated_fdr, f'Calibrated FDR:', calibrated_fdr)
                    print(f'Uncalibrated Power:', uncalibrated_power, f'Calibrated Power:', calibrated_power)
                    print('-'*50)
                '''

                round_interactions_df['uncalibrated_threshold'] = threshold_uncalibrated
                round_interactions_df['calibrated_threshold'] = threshold_calibrated
                round_interactions_df['uncalibrated_q_values'] = round_interactions_df.apply(lambda row: uncalibrated_q_values[row['pair']], axis=1)
                round_interactions_df['calibrated_q_values'] = round_interactions_df.apply(lambda row: calibrated_q_values[row['pair']], axis=1)

                record['uncalibrated_auc'] = uncalibrated_auc
                record['uncalibrated_fdr'] = uncalibrated_fdr
                record['uncalibrated_power'] = uncalibrated_power
                record['calibrated_auc'] = calibrated_auc
                record['calibrated_fdr'] = calibrated_fdr
                record['calibrated_power'] = calibrated_power

                fdr_power_records.append(record)
                attributions_dfs.append(round_attributions_df)
                interactions_dfs.append(round_interactions_df)

        attributions_df = pd.concat(attributions_dfs)
        interactions_df = pd.concat(interactions_dfs)
        fdr_power_df = pd.DataFrame(fdr_power_records)
        
        attributions_df.to_csv(attributions_df_path, index=False)
        interactions_df.to_csv(interactions_df_path, index=False)
        fdr_power_df.to_csv(fdr_power_df_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_fdr", type=float, default=0.2)
    args = parser.parse_args()
    
    fdr_control(args.data_dir, args.target_fdr)