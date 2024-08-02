import torch, math, copy, random
from itertools import combinations
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def get_torch_loaders(X, Y, batch_size, train_ratio=0.7):
    n = X.shape[0]
    n_train = math.ceil(n*train_ratio)

    X_train = X[:n_train, :]
    Y_train = Y[:n_train, :]
    X_val = X[n_train:, :]
    Y_val = Y[n_train:, :]

    loader_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train)), batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val)), batch_size, shuffle=False)

    return {"train": loader_train, "val": loader_val}, {"train": (X_train, Y_train), "val": (X_val, Y_val)}

def evaluate(net, data_loader, criterion, device):
    losses = []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = criterion(net(inputs), labels).cpu().data
        losses.append(loss)
    return torch.stack(losses).mean()

def train(
    net,
    data_loaders,
    optimizer,
    criterion=nn.MSELoss(reduction="mean"),
    nepochs=100,
    verbose=False,
    early_stopping=True,
    patience=5,
    decay_const=1e-4,
    device=torch.device("cpu"),
):
    if "val" not in data_loaders: early_stopping = False
    patience_counter = 0
    if verbose:
        print("starting to train")
        if early_stopping: print("early stopping enabled")
    def get_reg_loss():
        reg = 0
        for name, param in net.named_parameters():
            if "mlp" in name and "weight" in name:
                reg += torch.sum(torch.abs(param))
        return reg * decay_const

    best_net = None
    best_loss = float("inf")
    train_losses, val_losses, reg_losses = [], [], []
    for epoch in range(nepochs):
        running_loss = 0.0
        run_count = 0
        for i, data in enumerate(data_loaders["train"], 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()
            reg_loss = get_reg_loss()
            (loss + reg_loss).backward()
            optimizer.step()
            running_loss += loss.item()
            run_count += 1
        
        key = "val" if "val" in data_loaders else "train"
        train_loss = running_loss / run_count
        val_loss = evaluate(net, data_loaders[key], criterion, device)
        l1_reg, l2_reg = 0, 0
        reg_loss = get_reg_loss()
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        reg_losses.append(reg_loss.item())
        if verbose:
            print("[epoch %d, total %d] train loss: %.5f, val loss: %.5f"
                % (epoch + 1, nepochs, train_loss, val_loss))
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_net = copy.deepcopy(net)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience and early_stopping:
                if verbose:
                    print("early stopping!")
                break

    return best_net, optimizer, train_losses, val_losses, reg_losses

def data_dependent_threshold(W, fdr=0.10, offset=1):
    """
    Calculate data-dependent threshold given W statistics.

    Parameters
    ----------
    W : np.ndarray
        p-length numpy array of feature statistics OR (p, batch_length)
        shaped array.
    fdr : float
        desired level of false discovery rate control
    offset : int
        If offset = 0, control the modified FDR.
        If offset = 1 (default), controls the FDR exactly.
    Returns
    -------
    T : float or np.ndarray
        The data-dependent threshold. Either a float or a (batch_length,)
        dimensional array.
    """

    # Add dummy batch axis if necessary
    if len(W.shape) == 1:
        W = W.reshape(-1, 1)
    p = W.shape[0]
    batch = W.shape[1]

    # Sort W by absolute values
    ind = np.argsort(-1 * np.abs(W), axis=0)
    sorted_W = np.take_along_axis(W, ind, axis=0)

    # Calculate ratios
    negatives = np.cumsum(sorted_W <= 0, axis=0)
    positives = np.cumsum(sorted_W > 0, axis=0)
    positives[positives == 0] = 1  # Don't divide by 0
    ratios = (negatives + offset) / positives

    # Add zero as an option to prevent index errors
    # (zero means select everything strictly > 0)
    sorted_W = np.concatenate([sorted_W, np.zeros((1, batch))], axis=0)

    # Find maximum indexes satisfying FDR control
    # Adding np.arange is just a batching trick
    helper = (ratios <= fdr) + np.arange(0, p, 1).reshape(-1, 1) / p
    sorted_W[1:][helper < 1] = np.inf  # Never select values where the ratio > fdr
    T_inds = np.argmax(helper, axis=0) + 1
    more_inds = np.indices(T_inds.shape)

    # Find Ts
    acceptable = np.abs(sorted_W)[T_inds, more_inds][0]

    # Replace 0s with a very small value to ensure that
    # downstream you don't select W statistics == 0.
    # This value is the smallest abs value of nonzero W
    if np.sum(acceptable == 0) != 0:
        W_new = W.copy()
        W_new[W_new == 0] = np.abs(W).max()
        zero_replacement = np.abs(W_new).min(axis=0)
        acceptable[acceptable == 0] = zero_replacement[acceptable == 0]

    if batch == 1:
        acceptable = acceptable[0]

    return acceptable

def feat_import_power(sel_idx, gt_idx):
    if len(gt_idx) == 0:
        return 0
    true_sel = list(set(gt_idx).intersection(set(sel_idx)))
    return len(true_sel) / len(gt_idx)

def feat_import_fdp(sel_idx, gt_idx):
    if len(sel_idx) == 0:
        return 0
    false_sel = list(set(sel_idx).difference(set(gt_idx)))
    return len(false_sel) / len(sel_idx)

def pairwise_interaction_auc(interactions, ground_truth):
    strengths = []
    gt_binary_list = []
    for inter, strength in interactions:
        inter_set = set(inter)
        strengths.append(strength)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)
    auc = roc_auc_score(gt_binary_list, strengths)
    return auc

def get_interaction_type_idx(inter_indices_arr, p):
    TD_idx = []
    TT_idx = []
    DD_idx = []
    for idx, indices in enumerate(inter_indices_arr):
        if all(i < p for i in indices):
            TT_idx.append(idx)
        elif all(i >= p for i in indices):
            DD_idx.append(idx)
        else:
            TD_idx.append(idx)
    return TT_idx, TD_idx, DD_idx

def get_selected_interactions(interactions, p, q):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TT_idx, TD_idx, DD_idx = get_interaction_type_idx(inter_idices_arr, p)
    cutoff = np.inf
    max_score = -np.inf
    for idx, t in interactions:
        TT_count = (inter_score_arr[TT_idx] >= t).sum()
        TD_count = (inter_score_arr[TD_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        score = np.abs((TD_count-DD_count)) / max(1.0, TT_count)
        score = np.clip(score, 0, 1)
        if score > max_score:
            max_score = score
        else:
            score = max_score
        if (t < cutoff) and (score <= q):
            cutoff = t
        if score > q:
            break
    selected_interaction = []
    for idx, score in interactions:
        i, j = idx
        if score >= cutoff and (i < p) and (j < p):
            selected_interaction.append((idx, score))
    
    if cutoff == np.inf:
        cutoff = np.max(inter_score_arr) + 1e-6

    return selected_interaction, cutoff

def get_q_values(interactions, p):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TT_idx, TD_idx, DD_idx = get_interaction_type_idx(inter_idices_arr, p)
    q_values = {}
    max_score = -np.inf
    for idx, t in interactions:
        TT_count = (inter_score_arr[TT_idx] >= t).sum()
        TD_count = (inter_score_arr[TD_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        score = np.abs((TD_count-DD_count)) / max(1.0, TT_count)
        score = np.clip(score, 0, 1)
        if score > max_score:
            max_score = score
        else:
            score = max_score
        q_values[idx] = score

    return q_values

def get_3rd_interaction_type_idx(inter_indices_arr, p):
    D_idx = []
    DD_idx = []
    DDD_idx = []
    TTT_idx = []
    for idx, indices in enumerate(inter_indices_arr):
        if all(i < p for i in indices):
            TTT_idx.append(idx)
        elif all(i >= p for i in indices):
            DDD_idx.append(idx)
        elif sum(i<p for i in indices)==1:
            D_idx.append(idx)
        else:
            DD_idx.append(idx)
    return TTT_idx, D_idx, DD_idx, DDD_idx

def get_selected_3rd_interactions(interactions, p, q, abs_diff=True):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TTT_idx, D_idx, DD_idx, DDD_idx = get_3rd_interaction_type_idx(inter_idices_arr, p)
    cutoff = np.inf
    max_score = -np.inf
    for indices, t in interactions:
        TTT_count = (inter_score_arr[TTT_idx] >= t).sum()
        D_count = (inter_score_arr[D_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        DDD_count = (inter_score_arr[DDD_idx] >= t).sum()
        if abs_diff:
            score = abs(((D_count+DD_count+DDD_count)-
                2*(DD_count+DDD_count)+3*DDD_count))/ max(1.0, TTT_count)
        else:
            score = ((D_count+DD_count+DDD_count)-
                2*(DD_count+DDD_count)+3*DDD_count)/ max(1.0, TTT_count)
        score = np.clip(score, 0, 1)
        if score > max_score:
            max_score = score
        else:
            score = max_score
        if (t < cutoff) and (score <= q):
            cutoff = t
        if score > q: 
            break
    selected_interaction = []
    for idx, score in interactions:
        i, j, k = idx
        if score >= cutoff and (i < p) and (j < p) and (k < p):
            selected_interaction.append((idx, score))

    if cutoff == np.inf:
        cutoff = np.max(inter_score_arr) + 1e-6

    return selected_interaction, cutoff

def get_q_values_3rd(interactions, p):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TTT_idx, D_idx, DD_idx, DDD_idx = get_3rd_interaction_type_idx(inter_idices_arr, p)
    q_values = {}
    max_score = -np.inf
    for idx, t in interactions:
        TTT_count = (inter_score_arr[TTT_idx] >= t).sum()
        D_count = (inter_score_arr[D_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        DDD_count = (inter_score_arr[DDD_idx] >= t).sum()
        score = abs(((D_count+DD_count+DDD_count)-
            2*(DD_count+DDD_count)+3*DDD_count))/ max(1.0, TTT_count)
        score = np.clip(score, 0, 1)
        if score > max_score:
            max_score = score
        else:
            score = max_score
        q_values[idx] = score

    return q_values


def get_interaction_FDR_estimate(interactions, p, abs_diff=True, only_original=True):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TT_idx, TD_idx, DD_idx = get_interaction_type_idx(inter_idices_arr, p)
    fdr_indices = []
    fdr_values = []
    for indices, t in interactions:
        i, j = indices
        if only_original:
            if not(i < p and j < p): continue
        TT_count = (inter_score_arr[TT_idx] >= t).sum()
        TD_count = (inter_score_arr[TD_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        if abs_diff:
            fdr = min(1.0, abs(TD_count-DD_count) / max(1.0, TT_count))
        else:
            fdr = min(1.0, (TD_count-DD_count) / max(1.0, TT_count))
        fdr_values.append(fdr)
        fdr_indices.append(indices)
    # keep monotonic
    for i in range(1, len(fdr_values)):
        if fdr_values[i-1]>fdr_values[i]: fdr_values[i] = fdr_values[i-1]

    interaction_fdr = []
    for i in range(len(fdr_values)):
        interaction_fdr.append((fdr_indices[i], fdr_values[i]))
    interaction_fdr.sort(key=lambda x: x[1], reverse=False)

    return interaction_fdr

def get_gt_bins(interactions, ground_truth):
    """get binary label and the score"""
    scores = []
    gt_binaries = []
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    for inter, score in interactions:
        inter_set = set(inter)
        scores.append(score)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binaries.append(1)
        else:
            gt_binaries.append(0)
    return gt_binaries[::-1], scores[::-1]

def get_q_level_counts(scores, gt_binaries, step_size=1/100):
    scores = np.asarray(scores)
    gt_binaries = np.asarray(gt_binaries)
    q_val_arr = np.arange(0, 1+step_size, step_size)
    total_count_list = []
    gt_count_list = []
    for q_val in q_val_arr:
        sel_binaries = gt_binaries[np.where(scores <= q_val)[0]]
        total_count_list.append(len(sel_binaries))
        gt_count_list.append(sum(sel_binaries))
    return q_val_arr, total_count_list, gt_count_list


def get_interaction_fdp(sel_interactions, ground_truth):
    binaries, _ = get_gt_bins(sel_interactions, ground_truth)
    fdp = (len(binaries) - sum(binaries)) / max(1, len(binaries))
    return fdp

def get_interaction_power(sel_interactions, ground_truth, order=2):
    binaries, _ = get_gt_bins(sel_interactions, ground_truth)
    num_gt = 0
    all_gt = []
    for gt in ground_truth:
        curr_gt = list(combinations(gt, order))
        all_gt += curr_gt
    all_gt = set(all_gt)
    num_gt = len(all_gt)
    power = sum(binaries) / max(1, num_gt)
    return power

class CoxPHLoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        self.eps = eps
        super(CoxPHLoss, self).__init__()
    def forward(self, log_h, target):
        log_h = log_h.view(-1)
        target = target.view(-1)
        durations = torch.abs(target)
        events = (target<0).float()
        idx = durations.sort(descending=True)[1]
        events = events[idx]
        log_h = log_h[idx]
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(self.eps).log().add(gamma)
        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

def get_interaction_ranking_from_matrix(matrix):
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    p = matrix.shape[0]
    assert p%2 == 0
    p /= 2

    interaction_ranking = []
    def inter_func(i, j):
        score = np.abs(matrix[int(i), int(j)])
        interaction_ranking.append(((i, j), score))
    # original-original (i!=j)
    for comb in combinations(np.arange(p), 2): inter_func(*comb)
    # knockoff-knockoff (i!=j)
    for comb in combinations(np.arange(p, 2*p), 2): inter_func(*comb)
    # original-knockoff combinations
    for i in np.arange(p):
        for j in np.arange(p, 2*p):
            if (i != j-p): inter_func(i, j)
    # knockoff-original combinations
    for i in np.arange(p, 2*p):
        for j in np.arange(p):
            if (i != j+p): inter_func(i, j)
    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking