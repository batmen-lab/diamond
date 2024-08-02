import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def PlotScatterHelper(A, B, ax=None):
    """
    Plot the entries of a matrix A vs those of B
    :param A: n-by-p data matrix
    :param B: n-by-p data matrix
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.set_xlim([-0.2, 1.1])
    ax.set_ylim([-0.2, 1.1])

    for i in range(0,A.shape[0]-1):
        if i == 0:
            ax.scatter(A.diagonal(i),B.diagonal(i), alpha=0.2, label="diagonal pairs")
        else:
            ax.scatter(A.diagonal(i),B.diagonal(i), alpha=0.2)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    for i in range(0,A.shape[0]-1):
        meanValA = np.mean(A.diagonal(i))
        meanValB = np.mean(B.diagonal(i))
        ax.plot([meanValA, meanValA],lims, 'k-', alpha=0.2, zorder=0)
        if i==0:
            color = 'r-'
            alpha = 1
        else:
            color = 'k-'
            alpha = 0.2
        ax.plot(lims, [meanValB, meanValB], color, alpha=alpha, zorder=0)

    # Plot both limits against each other
    ax.plot(lims, lims, 'k-', dashes=[2,2], alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.legend()
    return ax

def ScatterCovariance(X, Xk):
    """ Plot the entries of Cov(Xk,Xk) vs Cov(X,X) and Cov(X,Xk) vs Cov(X,X)
    :param X: n-by-p matrix of original variables
    :param Xk: n-by-p matrix of knockoffs
    """
    # Create subplots
    fig, axarr = plt.subplots(1, 2, figsize=(14,7))

    # Originality
    XX = np.corrcoef(X.T)
    XkXk = np.corrcoef(Xk.T)
    PlotScatterHelper(XX, XkXk, ax=axarr[0])
    axarr[0].set_xlabel(r"$Cov(\mathbf{X},\mathbf{X})$")
    axarr[0].set_ylabel(r'$Cov(\tilde{\mathbf{X}},\tilde{\mathbf{X}})$')

    # Exchangeability
    p = X.shape[1]
    G = np.corrcoef(X.T, Xk.T)
    XX  = G[:p,:p]
    XXk = G[:p,p:(2*p)]
    PlotScatterHelper(XX, XXk, ax=axarr[1])
    axarr[1].set_xlabel(r'$Cov(\mathbf{X}, \mathbf{X})$')
    axarr[1].set_ylabel(r'$Cov(\mathbf{X}, \tilde{\mathbf{X}})$')
    fig.suptitle("Knockoff covariance plots")
    return fig

def ScatterPredict(X, C):
    fig = plt.figure(figsize=(7,7))
    min_val, max_val = min(X.min(), C.min()), max(X.max(), C.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="red")
    plt.plot(X.flatten(), C.flatten(), ".")
    plt.title("Scatter Predict Plot")
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    return fig

def get_importance_df(importances, p, import_gt=None):
    feat_import_df = {"value": [], "type": []}
    for i in range(importances.shape[0]):
        feat_import_df["value"].append(importances[i])
        if import_gt is not None:
            if i in import_gt:
                feat_import_df["type"].append("marginal features")
            elif i < p:
                feat_import_df["type"].append("irrelevant features")
            else:
                feat_import_df["type"].append("knockoff features")
        else:
            if i < p:
                feat_import_df["type"].append("original features")
            else:
                feat_import_df["type"].append("knockoff features")
    feat_import_df = pd.DataFrame(feat_import_df)
    return feat_import_df

def get_interaction_df(interactions, p, import_gt=None, inter_gt=None, feat_names=None, cutoff=None):
    feat_inter_df = {"value": [], "type": [], "i": [], "j": [], "name": [], "selected": []}
    for (i, j), score in interactions:
        feat_inter_df["value"].append(score)
        feat_inter_df["i"].append(i)
        feat_inter_df["j"].append(j)
        if feat_names is None:
            feat_inter_df["name"].append(f"{i} - {j}")
        else:
            if i < p:
                i_name = f"{feat_names[i]}"
            else:
                i_name = f"{feat_names[i-p]} (knockoff)"
            if j < p:
                j_name = f"{feat_names[j]}"
            else:
                j_name = f"{feat_names[j-p]} (knockoff)"
            feat_inter_df["name"].append(f"{i_name} - {j_name}")
        if cutoff is None:
            feat_inter_df["selected"].append(False)
        else:
            if score >= cutoff:
                feat_inter_df["selected"].append(True)
            else:
                feat_inter_df["selected"].append(False)
        if inter_gt is not None and import_gt is not None:
            if i < p and j < p:
                if any({i,j} <= gt for gt in inter_gt):
                    feat_inter_df["type"].append("Ground Truth Interactions")
                elif i in import_gt and j in import_gt:
                    feat_inter_df["type"].append("marginal-marginal")
                elif i in import_gt or j in import_gt:
                    feat_inter_df["type"].append("marginal-irrelevant")
                else:
                    feat_inter_df["type"].append("irrelevant-irrelevant")
            elif i < p or j < p:
                if i in import_gt or j in import_gt:
                    feat_inter_df["type"].append("marginal-knockoff")
                else:
                    feat_inter_df["type"].append("irrelevant-knockoff")
            else:
                feat_inter_df["type"].append("knockoff-knockoff")
        elif inter_gt is not None:
            if i < p and j < p:
                if any({i,j} <= gt for gt in inter_gt):
                    feat_inter_df["type"].append("Ground Truth Interactions")
                else:
                    feat_inter_df["type"].append("original-original")
            elif i < p or j < p:
                feat_inter_df["type"].append("original-knockoff")
            else:
                feat_inter_df["type"].append("knockoff-knockoff")
        else:
            if i < p and j < p:
                feat_inter_df["type"].append("original-original")
            elif i < p or j < p:
                feat_inter_df["type"].append("original-knockoff")
            else:
                feat_inter_df["type"].append("knockoff-knockoff")
    feat_inter_df = pd.DataFrame(feat_inter_df)
    return feat_inter_df


def get_3rd_interaction_df(interactions, p, import_gt=None, inter_gt=None, feat_names=None, cutoff=None):
    feat_inter_df = {"value": [], "type": [], "i": [], "j": [], "k": [], "name": [], "selected": []}
    for (i, j, k), score in interactions:
        feat_inter_df["value"].append(score)
        feat_inter_df["i"].append(i)
        feat_inter_df["j"].append(j)
        feat_inter_df["k"].append(k)
        if feat_names is None:
            feat_inter_df["name"].append(f"{i} - {j} - {k}")
        else:
            if i < p:
                i_name = f"{feat_names[i]}"
            else:
                i_name = f"{feat_names[i-p]} (knockoff)"
            if j < p:
                j_name = f"{feat_names[j]}"
            else:
                j_name = f"{feat_names[j-p]} (knockoff)"
            if k < p:
                k_name = f"{feat_names[k]}"
            else:
                k_name = f"{feat_names[k-p]} (knockoff)"
            feat_inter_df["name"].append(f"{i_name} - {j_name} - {k_name}")
        if cutoff is None:
            feat_inter_df["selected"].append(False)
        else:
            if score >= cutoff:
                feat_inter_df["selected"].append(True)
            else:
                feat_inter_df["selected"].append(False)
        if inter_gt is not None and import_gt is not None:
            if i < p and j < p and k < p:
                if any({i,j,k} <= gt for gt in inter_gt):
                    feat_inter_df["type"].append("Ground Truth Interactions")
                else:
                    feat_inter_df["type"].append("Original Interactions")
            elif i < p or j < p or k < p:
                feat_inter_df["type"].append("Original/Knockoff Interactions")
            else:
                feat_inter_df["type"].append("Knockoff Interactions")
        elif inter_gt is not None:
            if i < p and j < p and k < p:
                if any({i,j} <= gt for gt in inter_gt):
                    feat_inter_df["type"].append("Ground Truth Interactions")
                else:
                    feat_inter_df["type"].append("Original Interactions")
            elif i < p or j < p or k < p:
                feat_inter_df["type"].append("Original/Knockoff Interactions")
            else:
                feat_inter_df["type"].append("Knockoff Interactions")
        else:
            if i < p and j < p and k < p:
                feat_inter_df["type"].append("Original Interaction")
            elif i < p or j < p or k < p:
                feat_inter_df["type"].append("Original/Knockoff Interactions")
            else:
                feat_inter_df["type"].append("Knockoff Interactions")
    feat_inter_df = pd.DataFrame(feat_inter_df)
    return feat_inter_df

def get_simplified_interaction_df(interactions, p, inter_gt):
    feat_inter_df = {"value": [], "type": [], "i": [], "j": [], "Ground Truth": []}
    for (i, j), score in interactions:
        feat_inter_df["value"].append(score)
        feat_inter_df["i"].append(i)
        feat_inter_df["j"].append(j)
        if i < p and j < p:
            feat_inter_df["type"].append("Original-Original Interactions")
            if any({i,j} <= gt for gt in inter_gt):
                feat_inter_df["Ground Truth"].append(True)
            else:
                feat_inter_df["Ground Truth"].append(False)
        else:
            feat_inter_df["type"].append("Knockoff-* Interactions")
            feat_inter_df["Ground Truth"].append(False)
    return pd.DataFrame(feat_inter_df)

def get_simple_interaction_df(interactions, p, inter_gt):
    feat_inter_df = {"value": [], "type": [], "i": [], "j": [], "Ground Truth": []}
    for (i, j), score in interactions:
        feat_inter_df["value"].append(score)
        feat_inter_df["i"].append(i)
        feat_inter_df["j"].append(j)
        if any({i,j} <= gt for gt in inter_gt):
            feat_inter_df["type"].append("Ground truth\nInteractions")
            feat_inter_df["Ground Truth"].append(True)
        elif i < p and j < p:
            feat_inter_df["type"].append("Real-real\nInteractions")
            feat_inter_df["Ground Truth"].append(False)
        elif i < p and i < (j-p):
            feat_inter_df["type"].append("Real-Knockoff\nInteractions")
            feat_inter_df["Ground Truth"].append(False)
        elif j < p and (i-p) < j:
            feat_inter_df["type"].append("Knockoff-Real\nInteractions")
            feat_inter_df["Ground Truth"].append(False)
        else:
            feat_inter_df["type"].append("Knockoff-knockoff\nInteractions")
            feat_inter_df["Ground Truth"].append(False)
    return pd.DataFrame(feat_inter_df)


def get_3rd_order_interaction_df(interactions, p, inter_gt):
    feat_inter_df = {"value": [], "type": [], "i": [], "j": [], "k": []}
    for (i, j, k), score in interactions:
        feat_inter_df["value"].append(score)
        feat_inter_df["i"].append(i)
        feat_inter_df["j"].append(j)
        feat_inter_df["k"].append(k)
        if i < p and j < p and k < p:
            if any({i,j, k} <= gt for gt in inter_gt):
                feat_inter_df["type"].append("Ground Truth Interactions")
            else:
                feat_inter_df["type"].append("original-original-original")
        elif i>=p and j>=p and k>=p:
            feat_inter_df["type"].append("knockoff-knockoff-knockoff")
        else:
            feat_inter_df["type"].append("knockoff+original")
    feat_inter_df = pd.DataFrame(feat_inter_df)
    return feat_inter_df




def plot_knockoff_correlation(corr_arr, plot_url):
    assert len(corr_arr) == 4
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    for i, corr in enumerate(corr_arr):
        y = i // 2
        x = i % 2
        axes[y, x].plot(corr[0], corr[1], ".")
        axes[y, x].plot((min(corr[0]), max(corr[0])), (min(corr[1]), max(corr[1])), "-")

    if plot_url: fig.savefig(plot_url)

def knockoff_correlation(A1, tilde_A1, subset_dim=1000):
    if A1.shape[1] <= subset_dim:
        A = A1
        tilde_A = tilde_A1
    else:
        subset_indices = list(range(A1.shape[1]))
        np.random.shuffle(subset_indices)
        subset_indices = subset_indices[:subset_dim]

        A = A1[:, subset_indices]
        tilde_A = tilde_A1[:, subset_indices]
    print('A={}\ttilde_A={}'.format(A.shape, tilde_A.shape, ))

    ori_ll = ori_ur = np.dot(A.T, A)
    ori_ul = ori_lr = np.dot(A.T, A)
    np.fill_diagonal(ori_ur, 0)

    knockoff_ul = np.dot(A.T, A)
    knockoff_ur = np.dot(A.T, tilde_A)
    np.fill_diagonal(knockoff_ur, 0)
    knockoff_ll = np.dot(tilde_A.T, A)
    np.fill_diagonal(knockoff_ll, 0)
    knockoff_lr = np.dot(tilde_A.T, tilde_A)

    return [(ori_ul.flatten(), knockoff_ul.flatten()),
            (ori_ur.flatten(), knockoff_ur.flatten()),
            (ori_ll.flatten(), knockoff_ll.flatten()),
            (ori_lr.flatten(), knockoff_lr.flatten())]


def plot_training_results(train_losses, val_losses, reg_losses=None, figure_url=''):
    assert len(train_losses) == len(val_losses)

    fig = plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    if reg_losses is not None:
        plt.plot(reg_losses, label="reg")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.title(f"best val_loss={min(val_losses):.3f} best epoch={np.argmin(val_losses)}")

    if len(figure_url) > 0: fig.savefig(figure_url)
    else: plt.show()
    plt.close()

def plot_roc_curve_with_confidence(scores, labels, score_name, legend_name,
        mean_color="b", area_color="lightsteelblue", fig=None, ax=None):
    assert len(labels) == len(scores)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(scores)):
        curr_fpr, curr_tpr, _ = metrics.roc_curve(labels[i], scores[i])
        curr_roc_auc = metrics.auc(curr_fpr, curr_tpr)
        interp_tpr = np.interp(mean_fpr, curr_fpr, curr_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(curr_roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=mean_color,
        label=f"{legend_name} Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})",
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color=area_color,
        alpha=0.2,
        label=f"{legend_name} $\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve of {score_name}",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    return fig, mean_fpr, mean_tpr