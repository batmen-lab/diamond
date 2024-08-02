import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from adjustText import adjust_text

from fdr_control import prepare_interaction_dataframe


def plot_interaction_distribution(interactions_df: pd.DataFrame, save_dir: str) -> None:
    tab10 = sns.color_palette()
    for dataset in interactions_df['dataset'].unique():
        for seed in interactions_df['seed'].unique():
            interactions_df_func = interactions_df[(interactions_df['dataset'] == dataset) & (interactions_df['seed'] == seed)]
            for col in ['interaction', 'calibrated_interaction']:
                np.random.seed(0)
                plt.figure(figsize=(10, 10))
                
                # Plot all data except 'True'
                sns.stripplot(data=interactions_df_func[interactions_df_func['subtype'] != 'True'], 
                              y=col, 
                              x='type',
                              hue='type',
                              hue_order=['Original', 'Knockoff'],
                              palette={'Original': tab10[0], 'Knockoff': tab10[1]},
                              alpha=0.8)
                
                # Overlay the 'True' subtype
                sns.stripplot(data=interactions_df_func[interactions_df_func['subtype'] == 'True'], 
                              y=col, 
                              x='type',
                              hue='type',
                              hue_order=['Original', 'Knockoff'],
                              palette={'Original': tab10[3], 'Knockoff': tab10[3]},
                              marker='*',
                              s=10,
                              alpha=0.8)
                
                plt.title(f'{dataset} - {col}')
                plt.xlabel('')
                plt.ylabel('Interaction Score')
                cutoff = interactions_df_func['uncalibrated_threshold'].mean() if col == 'interaction' else interactions_df_func['calibrated_threshold'].mean()
                if cutoff > 0:
                    plt.axhline(y=cutoff, color='r', linestyle='--', label='Cutoff at target FDR')


                # Add legend for 'True' subtype and cutoff line
                handles, labels = plt.gca().get_legend_handles_labels()
                true_patch = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=tab10[3], label='Ground truth', markersize=12, alpha=0.8)
                handles.append(true_patch)
                labels.append('Ground truth')
                plt.legend(handles=handles, title='')

                plt.tight_layout()
                plt.savefig(f'{save_dir}/{dataset}_{seed}_{col}_stripplot.pdf')

                plt.figure(figsize=(10, 10))
                sns.ecdfplot(data=interactions_df_func[interactions_df_func['subtype'] != 'True'],
                            x=col, 
                            hue='type',
                            hue_order=['Original', 'Knockoff'],
                            palette={'Original': tab10[0], 'Knockoff': tab10[1]})
                plt.xlabel('Interaction Score')
                plt.ylabel('Cumulative Density')
                plt.ylim(-0.02, 1.02)
                plt.xscale('log')
                plt.tight_layout()
                plt.savefig(f'{save_dir}/{dataset}_{seed}_{col}_ecdf.pdf')

                plt.close('all')

def plot_attribution_distribution(attributions_df: pd.DataFrame, save_dir: str) -> None:
    tab10 = sns.color_palette()
    for dataset in attributions_df['dataset'].unique():
        for seed in attributions_df['seed'].unique():
            attributions_df_func = attributions_df[(attributions_df['dataset'] == dataset) & (attributions_df['seed'] == seed)]
            for col in ['attribution']:
                np.random.seed(0)
                plt.figure(figsize=(10, 10))
                
                # Plot all data except 'True'
                sns.stripplot(data=attributions_df_func[attributions_df_func['subtype'] != 'True'], 
                              y=col, 
                              x='type',
                              hue='type',
                              hue_order=['Original', 'Knockoff'],
                              palette={'Original': tab10[0], 'Knockoff': tab10[1],
                                       'True': tab10[3]},
                              alpha=0.8)
                
                # Overlay the 'True' subtype
                sns.stripplot(data=attributions_df_func[attributions_df_func['subtype'] == 'True'], 
                              y=col, 
                              x='type',
                              hue='type',
                              hue_order=['Original', 'Knockoff'],
                              palette={'Original': tab10[3], 'Knockoff': tab10[3],
                                       'True': tab10[3]},
                              marker='*',
                              s=10,
                              alpha=0.8)
                
                plt.title(f'{dataset} - {col}')
                plt.xlabel('')
                plt.ylabel('Attribution Score')
    
                plt.tight_layout()
                plt.savefig(f'{save_dir}/{dataset}_{seed}_{col}_stripplot.pdf')

                plt.close('all')

def plot_q_values(interactions_df: pd.DataFrame, save_dir: str) -> None:
    tab10 = sns.color_palette()
    original_df = interactions_df[interactions_df['type'] == 'Original']
    grouped = original_df.groupby(['dataset', 'pair']).agg({'uncalibrated_q_values': 'mean', 'calibrated_q_values': 'mean', 'name': 'first', 'subtype': 'first'}).reset_index()
    for dataset in grouped['dataset'].unique():
        grouped_dataset = grouped[grouped['dataset'] == dataset]
        if dataset == 'mortality':
            grouped_dataset['subtype'] = 'False'
        for col in ['uncalibrated_q_values', 'calibrated_q_values']:
            np.random.seed(0)
            plt.figure(figsize=(6, 10))

             # Plot false data
            false_data = grouped_dataset[grouped_dataset['subtype'] != 'True']
            false_data = false_data.sort_values(by=col, ascending=False)
            sns.stripplot(data=false_data,
                         y=col,
                         color = tab10[0],
                         s=10,
                         alpha=0.8,
                         jitter=0.2)
            
            # Plot true data
            true_data = grouped_dataset[grouped_dataset['subtype'] == 'True']
            true_data = true_data.sort_values(by=col, ascending=False)
            ax = sns.stripplot(data=true_data,
                        y=col,
                        color = tab10[3],
                        marker='*',
                        s=20,
                        alpha=0.8,
                        jitter=0.2)

            # Add text labels for top 10 q-values
            data = pd.concat([false_data, true_data])
            coordinates = []
            texts = []
            for collection in ax.collections:
                coordinates.extend(collection.get_offsets())
            top10 = data.sort_values(by=col, ascending=True).head(10)
            if col == 'calibrated_q_values':
                print(f'Top 10 {dataset} {col}')
                for i, row in top10.iterrows():
                    print(row['name'], row['calibrated_q_values'])
            for i, (x, y) in enumerate(coordinates):
                row = data.iloc[i]
                if row['pair'] in top10['pair'].values:
                    texts.append(plt.text(x, y, row['name'], fontsize=6, ha='center', va='center'))
            adjust_text(texts)
                        
            plt.title(f'{dataset} - {col}')
            plt.xlabel('')
            plt.ylabel('Q-value')
            plt.ylim(-0.02, 1.02)
            plt.axhline(y=0.2, color='r', linestyle='--', label='Target FDR')

                    
            # Add legend for 'True' subtype and cutoff line
            handles, labels = plt.gca().get_legend_handles_labels()
            true_patch = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=tab10[3], label='Ground truth', markersize=12, alpha=0.8)
            handles.append(true_patch)
            labels.append('Ground truth')
            plt.legend(handles=handles, title='')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{dataset}_{col}_stripplot.pdf')

            plt.close('all')
        
def plot_metric(results_df: pd.DataFrame, save_dir: str, metric2label: dict, simulation: bool = True) -> None:
    if simulation:
        order = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    else:
        order = results_df['dataset'].unique().tolist()
    for metric, label in metric2label.items():
        sns.set_theme(style='white')
        plt.figure(figsize=(10, 10))
        sns.barplot(data=results_df, 
                    x='dataset',
                    order=order,
                    y=metric, 
                    hue='dataset',
                    hue_order=order,
                    palette='mako',
                    errorbar='ci')
        plt.title(label)
        plt.xlabel('')
        plt.ylabel('')
        plt.ylim(0, 1.05)

        if 'fdr' in metric:
            plt.axhline(y=0.2, color='r', linestyle='--', label='Target FDR')

        annotate_bar_plot(plt.gca())

        plt.tight_layout()
        plt.savefig(f'{save_dir}/{metric}.pdf')

        plt.close('all')

def annotate_bar_plot(ax):
    is_log_scale = ax.get_yscale() == 'log'

    # Get the limits of the y-axis
    ylim = ax.get_ylim()
    log_range = np.log10(ylim[1]/ylim[0]) if is_log_scale else None

    # Iterate over each bar in the plot and annotate the height
    for p, err_bar in zip(ax.patches, ax.lines):
        bar_length = p.get_height()
        text_position = max(bar_length, err_bar.get_ydata()[1])

        if is_log_scale:
            # Calculate an offset that is proportional to the log scale
            log_offset = 0.01 * log_range
            y = text_position * (10 ** log_offset)
        else:
            # Use a fixed offset for linear scale
            offset = 0.01 * (ylim[1] - ylim[0])
            y = text_position + offset

        x = p.get_x() + p.get_width() / 2
        position = (x, y)
        ax.annotate(f"{bar_length:.2f}", xy=position, ha='center', va='center', fontsize=10, color='black')

def visualize_real(data_dir: str) -> None:
    # Create figure directory
    fig_dir = f'{data_dir}/figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Load data
    result_file = [f for f in os.listdir(data_dir) if f.endswith('.json') and 'seed1.' in f][0]
    with open(f'{data_dir}/{result_file}', 'r') as file:
        data = json.load(file)

    # Parse data from JSON
    seed = data['seed']
    dataset = data['dataset']
    num_features = len(data['attributions']) // 2
    feat_names = data['feat_names']
    interactions_df = prepare_interaction_dataframe(data)
    true_pairs = [set(pair) for pair in interactions_df[interactions_df['subtype'] == 'True']['pair']]

    # Load local attributions and interactions
    local_attributions_file = f'{data_dir}/{dataset}_seed{seed}_local_attributions.npy'
    local_interactions_file = f'{data_dir}/{dataset}_seed{seed}_local_interactions.npy'
    X_file = f'{data_dir}/{dataset}_seed{seed}_X.npy'

    if not all([os.path.exists(f) for f in [local_attributions_file, local_interactions_file, X_file]]):
        print('Missing data')
        return

    local_attributions = np.load(local_attributions_file)
    local_interactions = np.load(local_interactions_file)
    X = np.load(X_file)

    # Capitalize feature names
    feat_names = [name.upper() for name in feat_names]

    # Beeswarm plot for the original features
    sorted_feat = sorted(feat_names)
    feat2idx = {feat: idx for idx, feat in enumerate(feat_names)}
    order = list(map(feat2idx.get, sorted_feat))
    original_explanation = shap.Explanation(values=local_attributions[:, :num_features], 
                                    # data=X_test[:, :num_features],
                                    data=X[:, :num_features],
                                    feature_names=feat_names)
    shap.plots.beeswarm(original_explanation, 
                        max_display=len(feat_names), 
                        #order=original_explanation.abs.mean(0),
                        order=order,
                        plot_size=(20, 10),
                        show=False)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{dataset}_{seed}_original.pdf')
    plt.close()

    # Beeswarm plot for the knockoff features
    knockoff_explanation = shap.Explanation(values=local_attributions[:, num_features:], 
                                    # data=X_test[:, num_features:],
                                    data=X[:, num_features:],
                                    feature_names=feat_names)
    shap.plots.beeswarm(knockoff_explanation, 
                        max_display=len(feat_names), 
                        # order=original_explanation.abs.mean(0),
                        order = order,
                        plot_size=(20, 10),
                        show=False)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{dataset}_{seed}_knockoff.pdf')
    plt.close()
    
    # Plot dependency plots for top 5 and (if applicable) true interactions
    model_type = 'nn' if 'nn' in data_dir else 'lightgbm' if 'lightgbm' in data_dir else 'xgboost'
    names = []
    if dataset == 'enhancer':
        true_names = [(feat_names[i], feat_names[j]) for i, j in true_pairs]
        if model_type == 'nn':
            names = [('wt_ZLD', 'twi'), ('bcd', 'twi'), ('kr', 'twi'), ('sna', 'twi'), ('twi', 'z2')]
        elif model_type == 'lightgbm':
            names = [('H3K18ac', 'med2'), ('gt2', 'twi'), ('med2', 'twi'), ('D1', 'twi'), ('kr', 'twi')]
        elif model_type == 'xgboost':
            names = [('kr', 'twi'), ('D1', 'twi'), ('dl3', 'kni'), ('wt_ZLD', 'twi'), ('wt_ZLD', 'gt2')]
        names = list(set(names) | set(true_names))
    elif dataset == 'mortality':
        if model_type == 'nn':
            names = [('BUN', 'Sedimentation rate'), ('platelets_isNormal', 'Sedimentation rate'), ('BUN', 'platelets_isNormal'), ('monocytes', 'Sedimentation rate'), ('BUN', 'urine_pH'), 
                     ('BUN', 'monocytes'), ('Sex', 'Sedimentation rate'), ('BUN', 'creatinine'), ('BUN', 'potassium'), ('urine_hematest_isLarge', 'Sedimentation rate')]
    elif dataset == 'diabetes':
        names = [('bmi', 's5')]
    elif dataset == 'cal_housing':
        if model_type == 'nn':
            names = [('Latitude', 'Longitude'),  ('HouseAge', 'AveBedrms'), ('AveBedrms', 'Latitude'), ('AveBedrms', 'AveOccup'), ('AveRooms', 'Latitude')]
        elif model_type == 'lightgbm':
            names = [('Latitude', 'Longitude'), ('HouseAge', 'Longitude'), ('HouseAge', 'AveOccup'), ('HouseAge', 'Population'), ('MedInc', 'AveBedrms')]
        elif model_type == 'xgboost':
            names = [('Latitude', 'Longitude'), ('HouseAge', 'AveOccup'), ('MedInc', 'AveBedrms'), ('MedInc', 'Population'), ('MedInc', 'AveOccup')]
    elif dataset == 'bike_sharing':
        if model_type == 'nn':
            names = [('temp', 'yr'), ('hr==8', 'weekday==0'), ('workingday', 'atemp'), ('atemp', 'weekday==6'), ('atemp', 'yr==1'), ('workingday', 'yr==0'), ('hum', 'yr==0'), ('workingday', 'temp'), ('temp', 'yr==1')]

    names = [(name[0].upper(), name[1].upper()) for name in names]
    for name in names:
        original_explanation = shap.Explanation(values=local_interactions[:, :num_features, np.where(np.array(feat_names) == name[1])[0][0]],
                                data=X[:, :num_features],
                                feature_names=feat_names)
        shap.plots.scatter(shap_values=original_explanation[:, name[0]], 
                            color=original_explanation[:, name[1]], 
                            hist=False,
                            show=False)
        plt.ylabel(f'Interaction between {name[0]} and {name[1]}')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{dataset}_{seed}_{name[0]}_{name[1]}_vintv.pdf')


        original_explanation = shap.Explanation(values=local_interactions[:, :num_features, np.where(np.array(feat_names) == name[1])[0][0]],
                                data=local_attributions[:, :num_features],
                                feature_names=feat_names)
        shap.plots.scatter(shap_values=original_explanation[:, name[0]], 
                            color=original_explanation[:, name[1]], 
                            hist=False,
                            show=False)
        plt.ylabel(f'Interaction between {name[0]} and {name[1]}')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{dataset}_{seed}_{name[0]}_{name[1]}_ainta.pdf')

        original_explanation = shap.Explanation(values=local_attributions[:, :num_features],
                            data=X[:, :num_features],
                            feature_names=feat_names)
        shap.plots.scatter(shap_values=original_explanation[:, name[0]],
                            color=original_explanation[:, name[1]],
                            hist=False,
                            show=False)
        plt.ylabel(f'Attribution of {name[0]}')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{dataset}_{seed}_{name[0]}_{name[1]}_vav.pdf')

        
        plt.close('all')


def visualize_results(data_dir: str) -> None:
    # Visualize attribution and interaction scores for real data
    if 'simulation' not in data_dir:
        visualize_real(data_dir)

    # Set output directory
    output_dir = f'{data_dir}/fdr_control'
    fig_dir = f'{output_dir}/figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    attributions_df_path = f'{output_dir}/attributions.csv'
    interactions_df_path = f'{output_dir}/interactions.csv'
    fdr_power_df_path = f'{output_dir}/fdr_power.csv'
    
    if all([os.path.exists(f) for f in [interactions_df_path, fdr_power_df_path]]):
        print('Loaded existing data')
        interactions_df = pd.read_csv(interactions_df_path)
        fdr_power_df = pd.read_csv(fdr_power_df_path)

        metric2label = {
            'uncalibrated_auc': 'Uncalibrated AUC',
            'uncalibrated_fdr': 'Uncalibrated FDR',
            'uncalibrated_power': 'Uncalibrated Power',
            'calibrated_auc': 'Calibrated AUC',
            'calibrated_fdr': 'Calibrated FDR',
            'calibrated_power': 'Calibrated Power'
        }
        simulation = True if 'simulation' in data_dir else False
        if simulation:
            plot_metric(fdr_power_df, fig_dir, metric2label, simulation=simulation)
        plot_q_values(interactions_df, fig_dir)
        plot_interaction_distribution(interactions_df, fig_dir)
        if os.path.exists(attributions_df_path):
            attributions_df = pd.read_csv(attributions_df_path)
            attributions_df['subtype'] = attributions_df['subtype'].astype(str)
            plot_attribution_distribution(attributions_df, fig_dir)
    else:
        print('No data found')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    visualize_results(args.data_dir)