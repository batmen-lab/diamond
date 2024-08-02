import torch, itertools, math
from tqdm import tqdm
import numpy as np
from torch import nn
if torch.__version__.startswith("2"):
    from torch import vmap
    from torch.func import hessian
else:
    from functorch import vmap, hessian
from torch.autograd import grad
from .model_utils import get_selected_interactions
# from .shapiq.games import PytorchMetaGame, MachineLearningGame
# from .shapiq.approximators import SHAPIQEstimator

class DeepPINK(torch.nn.Module):
    
        def __init__(
                self, p, hidden_dims=[64], use_Z_weight=True, normalize_Z_weight=True
            ):
                super(DeepPINK, self).__init__()
                self.p = p
                self.hidden_dims = [self.p if use_Z_weight else self.p*2] + hidden_dims + [1]
                self.activation = torch.nn.ELU()
                self.use_Z_weight = use_Z_weight
                if self.use_Z_weight:
                    self.Z_weight = nn.Parameter(torch.ones(2 * p))
                else:
                    assert not normalize_Z_weight
                self.normalize_Z_weight = normalize_Z_weight
                # Create MLP layer layers
                mlp_layers = []
                for i in range(len(self.hidden_dims) - 1):
                    mlp_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                    if i+1 == len(self.hidden_dims) - 1: continue
                    mlp_layers.append(torch.nn.ELU())
                self.mlp = nn.Sequential(*mlp_layers)

        def _fetch_Z_weight(self):
            Z = self.Z_weight
            if self.normalize_Z_weight:
                normalizer = torch.abs(self.Z_weight[:self.p]) + \
                            torch.abs(self.Z_weight[self.p:])
                Z = torch.cat([torch.abs(self.Z_weight[:self.p]) / normalizer, 
                    torch.abs(self.Z_weight[self.p:]) / normalizer], dim=0)
            return Z

        def forward(self, X):
            if self.use_Z_weight:
                X_pink = self._fetch_Z_weight().unsqueeze(dim=0) * X
                X_mlp = X_pink[:, :self.p] + X_pink[:, self.p:]
            else:
                X_mlp = X
            return self.mlp(X_mlp)

        def _get_W(self):
            with torch.no_grad():
                # Calculate weights from MLP
                layers = list(self.mlp.named_children())
                W = None
                for layer in layers:
                    if isinstance(layer[1], torch.nn.Linear):
                        weight = layer[1].weight.cpu().detach().numpy().T
                        W = weight if W is None else np.dot(W, weight)
                W = W.squeeze(-1)
                return W

        def global_feature_importances(self):
            with torch.no_grad():
                # Calculate weights from MLP
                W = self._get_W()
                if self.use_Z_weight:
                    # Multiply by Z weights
                    Z = self._fetch_Z_weight().cpu().numpy()
                    feature_imp = Z[:self.p] * W
                    knockoff_imp = Z[self.p:] * W
                    return np.concatenate([feature_imp, knockoff_imp])
                else:
                    return W

        def get_weights(self):
            weights = []
            for name, param in self.named_parameters():
                if "mlp" in name and "weight" in name:
                    weights.append(param.cpu().detach().numpy())
            return weights

        def global_feature_interactions(self):
            with torch.no_grad():
                weights = self.get_weights()
                w_input = weights[0]
                w_later = weights[-1]
                for i in range(len(weights)-2, 0, -1):
                    w_later = np.matmul(w_later, weights[i])
                if self.use_Z_weight: Z = self._fetch_Z_weight().cpu().numpy()
                else: Z = np.ones(self.p*2)
                attributions = np.zeros((1, self.p*2))
                interactions = np.zeros((1, self.p*2, self.p*2))
                def inter_func(i, j):
                    w_input_i = Z[i]*w_input[:, i%w_input.shape[1]]
                    w_input_j = Z[j]*w_input[:, j%w_input.shape[1]]
                    attributions[0, i] = np.abs((w_input_i*w_later).sum())
                    attributions[0, j] = np.abs((w_input_j*w_later).sum())
                    interactions[0, i, j] =  np.abs((np.multiply(w_input_i, w_input_j)*w_later).sum())

                '''
                # original-original (i!=j)
                for comb in itertools.combinations(np.arange(self.p), 2): inter_func(*comb)
                # knockoff-knockoff (i!=j)
                for comb in itertools.combinations(np.arange(self.p, 2*self.p), 2): inter_func(*comb)
                # original-knockoff combinations
                for i in np.arange(self.p):
                    for j in np.arange(self.p, 2*self.p):
                        if (i != j-self.p): inter_func(i, j)
                # knockoff-original combinations
                for i in np.arange(self.p, 2*self.p):
                    for j in np.arange(self.p):
                        if (i != j+self.p): inter_func(i, j)
                '''
                for i, j in itertools.product(np.arange(self.p*2), repeat=2):
                    inter_func(i, j)

                return attributions, interactions

        def global_3rd_order_interactions(self):
            with torch.no_grad():
                weights = self.get_weights()
                w_input = weights[0]
                w_later = weights[-1]
                for i in range(len(weights)-2, 0, -1):
                    w_later = np.matmul(w_later, weights[i])
                if self.use_Z_weight: Z = self._fetch_Z_weight().cpu().numpy()
                else: Z = np.ones(self.p*2)
                attributions = np.zeros((1, self.p*2))
                interactions_2nd = np.zeros((1, self.p*2, self.p*2))
                interactions_3rd = np.zeros((1, self.p*2, self.p*2, self.p*2))
                
                def inter_func(i, j, k):
                    w_input_i = Z[i]*w_input[:, i%w_input.shape[1]]
                    w_input_j = Z[j]*w_input[:, j%w_input.shape[1]]
                    w_input_k = Z[k]*w_input[:, k%w_input.shape[1]]
                    attributions[0, i] = np.abs((w_input_i*w_later).sum())
                    attributions[0, j] = np.abs((w_input_j*w_later).sum())
                    attributions[0, k] = np.abs((w_input_k*w_later).sum())
                    interactions_2nd[0, i, j] = np.abs((np.multiply(w_input_i, w_input_j)*w_later).sum())
                    interactions_2nd[0, i, k] = np.abs((np.multiply(w_input_i, w_input_k)*w_later).sum())
                    interactions_2nd[0, j, k] = np.abs((np.multiply(w_input_j, w_input_k)*w_later).sum())
                    interactions_3rd[0, i, j, k] = np.abs((np.multiply(np.multiply(w_input_i, w_input_j), w_input_k)*w_later).sum())

                '''
                # original-original-original (i!=j!=k)
                for comb in itertools.combinations(np.arange(self.p), 3): inter_func(*comb)
                # knockoff-knockoff-knockoff (i!=j!=k)
                for comb in itertools.combinations(np.arange(self.p, 2*self.p), 3): inter_func(*comb)
                # knockoff-original-original combinations
                for i in np.arange(self.p, 2*self.p):
                    for j, k in itertools.combinations(np.arange(self.p), 2): 
                        if not (((i-self.p)==j) or ((i-self.p)==k)): 
                            inter_func(i, j, k)
                # original-knockof-knockoff combination
                for i in np.arange(self.p):
                    for j, k in itertools.combinations(np.arange(self.p, 2*self.p), 2): 
                        if not ((i==(j-self.p)) and (i==(k-self.p))): 
                            inter_func(i, j, k)
                '''
                for i, j, k in itertools.product(np.arange(self.p*2), repeat=3):
                    inter_func(i, j, k)
                
                return attributions, interactions_2nd, interactions_3rd


        def global_3rd_order_interactions_selected(self, indices_list, eps=1e-8, normalize=False, calibrate=True, calibrate_option=1):
            with torch.no_grad():
                weights = self.get_weights()
                w_input = weights[0]
                w_later = weights[-1]
                for i in range(len(weights)-2, 0, -1):
                    w_later = np.matmul(w_later, weights[i])
                if self.use_Z_weight: Z = self._fetch_Z_weight().cpu().numpy()
                else: Z = np.ones(self.p*2)
                interaction_ranking = []
                def inter_func(i, j, k):
                    w_input_i = (Z[i]*w_input[:, i%w_input.shape[1]])
                    w_input_j = (Z[j]*w_input[:, j%w_input.shape[1]])
                    w_input_k = (Z[k]*w_input[:, k%w_input.shape[1]])
                    inter_ijk = np.abs((np.multiply(np.multiply(w_input_i, w_input_j), w_input_k)*w_later).sum())
                    
                    if calibrate:
                        import_i = np.abs((w_input_i*w_later).sum())
                        import_j = np.abs((w_input_j*w_later).sum())
                        import_k = np.abs((w_input_k*w_later).sum())
                        if calibrate_option==1:
                            inter_ij = np.abs((np.multiply(w_input_i, w_input_j)*w_later).sum())# / (np.sqrt(import_i*import_j))
                            inter_ik = np.abs((np.multiply(w_input_i, w_input_k)*w_later).sum())# / (np.sqrt(import_i*import_k))
                            inter_jk = np.abs((np.multiply(w_input_j, w_input_k)*w_later).sum())# / (np.sqrt(import_j*import_k))
                            inter_ijk =  inter_ijk / ((inter_ij*inter_ik*inter_jk)**(1/3) * (import_k * import_j * import_i))**(1/2)
                        elif calibrate_option==2:
                            inter_ij = np.abs((np.multiply(w_input_i, w_input_j)*w_later).sum())# / (np.sqrt(import_i*import_j))
                            inter_ik = np.abs((np.multiply(w_input_i, w_input_k)*w_later).sum())# / (np.sqrt(import_i*import_k))
                            inter_jk = np.abs((np.multiply(w_input_j, w_input_k)*w_later).sum())# / (np.sqrt(import_j*import_k))
                            inter_ijk = inter_ijk / (inter_ij * import_k)**(1/2)
                            
                        # inter_ijk =  inter_ijk * (inter_ij*inter_ik*inter_jk)**(1/3) / (import_k*import_j*import_i)**(1/3)

                        # inter_ijk = inter_ijk / ((import_k * import_j * import_i) ** (1/3))
                        
                        
                        # inter_ijk =  (((inter_ijk / ((inter_ij * import_k)**(1/2)))) * 
                        #     (inter_ijk / ((inter_ik * import_j)**(1/2))) * 
                        #     (inter_ijk / ((inter_jk * import_i)**(1/2))))**(1/3)

                        # inter_ijk =  (((inter_ijk / ((inter_ij * import_k)**(1/2)))) +
                        #     (inter_ijk / ((inter_ik * import_j)**(1/2))) +
                        #     (inter_ijk / ((inter_jk * import_i)**(1/2))))/3

                        # inter_ijk =  max(((inter_ijk / ((inter_ij * import_k)**(1/2)))),
                        #     (inter_ijk / ((inter_ik * import_j)**(1/2))), 
                        #     (inter_ijk / ((inter_jk * import_i)**(1/2))))

                        # inter_ijk =  inter_ijk / ((inter_ij * import_k)**(1/2)*(inter_ik*import_j)**(1/2)*(inter_jk*import_i)**(1/2))**(1/3)
                        
                        # inter_ijk =  inter_ijk / max((inter_ij * import_k)**(1/2), (inter_ik*import_j)**(1/2), (inter_jk*import_i)**(1/2))
                        
                        # inter_ijk =  inter_ijk / (max(inter_ij, inter_ik, inter_jk) * max(import_k, import_j, import_i))**(1/2)



                    strength = np.abs(inter_ijk.mean())
                    interaction_ranking.append(((i, j, k), strength))

                for indices in indices_list:
                    inter_func(*indices)
                interaction_ranking.sort(key=lambda x: x[1], reverse=True)
                return interaction_ranking
                    


        def local_feature_importances(self, X, baseline, steps=100, expected=False):
            """expected: True --> expected hessian, False --> integrated hessian"""
            device = next(self.parameters()).device
            if not torch.is_tensor(X): X = torch.Tensor(X).to(device)
            if not torch.is_tensor(baseline): baseline = torch.Tensor(baseline).to(device)
            if len(X.shape) == 1: X = X.unsqueeze(0)
            if expected and len(baseline.shape)==1: baseline = baseline.unsqueeze(0)
            elif not expected and len(baseline.shape)==2: baseline = baseline.mean(0)
            X.requires_grad = True
            if expected:
                alpha_list = np.random.rand(steps+1)
                rand_idx_list = np.random.choice(baseline.shape[0], steps+1)
                scaled_X = torch.cat([baseline[r]+alpha_list[i]*(X.unsqueeze(1)-baseline[r])
                    for i, r in enumerate(rand_idx_list)], dim=1)
                delta_tensor = (X.unsqueeze(1)-baseline[rand_idx_list])
            else:    
                scaled_X = torch.cat([baseline+(float(i)/(steps))*(X.unsqueeze(1)-baseline)
                    for i in range(steps+1)], dim=1)
                delta_tensor = X.unsqueeze(1)-baseline

            grad_tensor = torch.zeros(scaled_X.shape).to(device)
            for i in range(steps+1):
                particular_slice = scaled_X[:,i]
                batch_output = self(particular_slice)
                grad_tensor[:,i,:] = grad(outputs=batch_output, inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(device),
                    create_graph=True)[0]
            attributions = torch.abs((delta_tensor * grad_tensor).sum(axis=1))
            return attributions.mean(axis=0).detach().cpu().numpy()

        def _local_feature_interactions_helper(self, X, baseline, steps=100, expected=False):
            # setup X
            device = next(self.parameters()).device
            if not torch.is_tensor(X): X = torch.Tensor(X).to(device)
            if not torch.is_tensor(baseline): baseline = torch.Tensor(baseline).to(device)
            if len(X.shape) == 1: X = X.unsqueeze(0)
            if expected and len(baseline.shape)==1: baseline = baseline.unsqueeze(0)
            elif not expected and len(baseline.shape)==2: baseline = baseline.mean(0)
            X.requires_grad = True
            # Setup delta tensor
            if expected:
                alpha_list = np.random.rand(steps+1)
                rand_idx_list = np.random.choice(baseline.shape[0], steps+1)
                scaled_X = torch.cat([baseline[r]+alpha_list[i]*(X.unsqueeze(1)-baseline[r])
                    for i, r in enumerate(rand_idx_list)], dim=1)
                grad_delta_tensor = (X.unsqueeze(1)-baseline[rand_idx_list]).unsqueeze(2)
                hess_delta_tensor = torch.stack([torch.bmm(grad_delta_tensor[i].permute(0, 2, 1), 
                    grad_delta_tensor[i]) for i in range(X.shape[0])])
                grad_delta_tensor = grad_delta_tensor.squeeze(2)                
            else:
                scaled_X = torch.cat([baseline+(float(i)/(steps))*(X.unsqueeze(1)-baseline)
                    for i in range(steps+1)], dim=1)
                alpha_list = [float(i)/steps for i in range(steps+1)]
                grad_delta_tensor = X.unsqueeze(1)-baseline
                hess_delta_tensor = torch.bmm(grad_delta_tensor.permute(0, 2, 1), grad_delta_tensor).unsqueeze(1)
            # Compute integrated gradient & hessian
            grad_tensor = torch.zeros(scaled_X.shape).to(device)
            hess_tensor = torch.zeros((*grad_tensor.shape, X.shape[1])).to(device)
            for i in range(steps+1):
                particular_slice = scaled_X[:,i]
                batch_output = self(particular_slice)
                grad_tensor[:,i,:] = grad(outputs=batch_output, inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(device),
                    create_graph=True)[0]
                hess_tensor[:, i, :, :] = vmap(hessian(self))(particular_slice).squeeze()*alpha_list[i]

            attributions = torch.abs((grad_delta_tensor*grad_tensor).sum(axis=1))
            interactions = torch.abs((hess_delta_tensor*hess_tensor).sum(axis=1))
            return attributions.detach().cpu().numpy(), interactions.detach().cpu().numpy()

        def local_feature_interactions(
            self, X, baseline, steps=100, eps=1e-8, batch_size=250,
            expected=True, normalize=False, calibrate=True):
            n = X.shape[0]
            attributions_list = []
            interactions_list = []
            for i in tqdm(range(math.ceil(n/batch_size))):
                curr_attributions, curr_interactions = self._local_feature_interactions_helper(
                    X[i*batch_size:(i+1)*batch_size], baseline, steps=steps, expected=expected)
                attributions_list.append(curr_attributions)            
                interactions_list.append(curr_interactions)
            import_arr = np.concatenate(attributions_list, axis=0)
            inter_arr = np.concatenate(interactions_list, axis=0)
            if calibrate:
                denominator_arr = np.sqrt(import_arr[..., None] * import_arr[:, None, :])
                inter_arr /= (denominator_arr+eps)
            inter_arr = inter_arr.mean(0)

            interaction_ranking = []
            def inter_func(i, j):
                score = np.abs(inter_arr[i, j])
                interaction_ranking.append(((i, j), score))
            # original-original (i!=j)
            for comb in itertools.combinations(np.arange(self.p), 2): inter_func(*comb)
            # knockoff-knockoff (i!=j)
            for comb in itertools.combinations(np.arange(self.p, 2*self.p), 2): inter_func(*comb)
            # original-knockoff combinations
            for i in np.arange(self.p):
                for j in np.arange(self.p, 2*self.p):
                    if (i != j-self.p): inter_func(i, j)
            # knockoff-original combinations
            for i in np.arange(self.p, 2*self.p):
                for j in np.arange(self.p):
                    if (i != j+self.p): inter_func(i, j)
            interaction_ranking.sort(key=lambda x: x[1], reverse=True)

            if normalize:
                max_score = interaction_ranking[0][1]
                min_score = interaction_ranking[-1][1]
                interaction_ranking = [((i, j), (s-min_score)/(max_score-min_score)) for (i, j), s in interaction_ranking]

            return interaction_ranking

        def hessian_feature_interactions(self, X, calibrate=True, normalize=False, eps=1e-8, noise=0.0):
            if noise > 0:
                sigma = (X.max() - X.min())*noise
                X += np.random.normal(loc=0, scale=sigma, size=X.shape)
            device = next(self.parameters()).device
            batch_X = torch.Tensor(X).to(device)
            batch_X.requires_grad = True
            batch_output = self(batch_X)
            batch_grad = np.abs(grad(outputs=batch_output, inputs=batch_X,
                grad_outputs=torch.ones_like(batch_output).to(device),
                create_graph=True)[0].cpu().detach().numpy())
            batch_tensor = vmap(hessian(self))(batch_X).squeeze().cpu().detach().numpy()
            if calibrate:
                denominator_arr = np.sqrt(batch_grad[..., None] * batch_grad[:, None, :])
                batch_tensor /= (denominator_arr+eps)
            batch_tensor = np.abs(batch_tensor.mean(0))

            interaction_ranking = []
            def inter_func(i, j):
                score = np.abs(batch_tensor[i, j])
                interaction_ranking.append(((i, j), score))
            # original-original (i!=j)
            for comb in itertools.combinations(np.arange(self.p), 2): inter_func(*comb)
            # knockoff-knockoff (i!=j)
            for comb in itertools.combinations(np.arange(self.p, 2*self.p), 2): inter_func(*comb)
            # original-knockoff combinations
            for i in np.arange(self.p):
                for j in np.arange(self.p, 2*self.p):
                    if (i != j-self.p): inter_func(i, j)
            # knockoff-original combinations
            for i in np.arange(self.p, 2*self.p):
                for j in np.arange(self.p):
                    if (i != j+self.p): inter_func(i, j)
            interaction_ranking.sort(key=lambda x: x[1], reverse=True)

            if normalize:
                max_score = interaction_ranking[0][1]
                min_score = interaction_ranking[-1][1]
                interaction_ranking = [((i, j), (s-min_score)/(max_score-min_score)) for (i, j), s in interaction_ranking]

            return interaction_ranking

        def hessian_3rd_order_interactions(self, X, calibrate=True, normalize=False, eps=1e-8, noise=0.0, batch_size=500):
            if noise > 0:
                sigma = (X.max() - X.min())*noise
                X += np.random.normal(loc=0, scale=sigma, size=X.shape)
            device = next(self.parameters()).device

            def compute_grad(batch_X):
                batch_X.requires_grad = True
                batch_output = self(batch_X)
                return grad(outputs=batch_output, inputs=batch_X,
                grad_outputs=torch.ones_like(batch_output).to(device),
                create_graph=True)[0]

            n = X.shape[0]
            batch_grad_list = []
            batch_hess_list = []
            batch_3rd_list = []
            for i in tqdm(range(math.ceil(n/batch_size))):
                batch_X = torch.Tensor(X[i*batch_size:(i+1)*batch_size]).to(device)
                batch_grad = compute_grad(batch_X).cpu().detach().numpy()
                batch_hess = vmap(hessian(self))(batch_X).squeeze().cpu().detach().numpy()
                batch_3rd = vmap(hessian(compute_grad))(batch_X).squeeze().cpu().detach().numpy()
                batch_grad_list.append(batch_grad)
                batch_hess_list.append(batch_hess)
                batch_3rd_list.append(batch_3rd)

            batch_grad = np.concatenate(batch_grad_list, axis=0)
            batch_hess = np.concatenate(batch_hess_list, axis=0)
            batch_3rd = np.concatenate(batch_3rd_list, axis=0)

            if calibrate:
                print(f"Compute Numerator")
                batch_1st = np.abs(batch_grad)
                batch_2nd = np.abs(batch_hess / np.sqrt(batch_1st[..., None] * batch_1st[:, None, :]))
                batch_numerator = np.zeros((n, self.p*2, self.p*2, self.p*2))
                for i, j, k in tqdm(list(itertools.combinations_with_replacement(np.arange(self.p*2), 3))):
                    curr_numerator = (batch_2nd[:, i, j] * batch_2nd[:, i, k] * batch_2nd[:, j, k])**(1/3)
                    for comb in list(itertools.permutations([i, j, k], 3)):
                        batch_numerator[:, comb[0], comb[1], comb[2]] = curr_numerator
                print(f"Compute Denominator")
                batch_denominator = ((batch_1st[..., None] * batch_1st[:, None, :])[..., None] * batch_1st[:, None, None, :])**(1/3)
                batch_3rd *= (batch_numerator / batch_denominator)

            inter_scores = np.abs(batch_3rd.mean(0))
            interaction_ranking = []
            def inter_func(i, j, k): interaction_ranking.append(((i, j, k), inter_scores[i, j, k]))

            # original-original-original (i!=j!=k)
            for comb in itertools.combinations(np.arange(self.p), 3): inter_func(*comb)
            # knockoff-knockoff-knockoff (i!=j!=k)
            for comb in itertools.combinations(np.arange(self.p, 2*self.p), 3): inter_func(*comb)
            # knockoff-original-original combinations
            for i in np.arange(self.p, 2*self.p):
                for j, k in itertools.combinations_with_replacement(np.arange(self.p), 2): 
                    if not ((i-self.p)==j==k): 
                        for comb in itertools.permutations([i, j, k], 3): 
                            inter_func(*comb)
            # original-knockof-knockoff combination
            for i in np.arange(self.p):
                for j, k in itertools.combinations_with_replacement(np.arange(self.p, 2*self.p), 2): 
                    if not (i==(j-self.p)==(k-self.p)): 
                        for comb in itertools.permutations([i, j, k], 3):
                            inter_func(*comb)
            
            interaction_ranking.sort(key=lambda x: x[1], reverse=True)
            if normalize:
                max_score = interaction_ranking[0][1]
                min_score = interaction_ranking[-1][1]
                interaction_ranking = [((i, j), (s-min_score)/(max_score-min_score)) for (i, j), s in interaction_ranking]
            return interaction_ranking
            


        def paraACE_feature_interactions(self, X, normalize=False, eps=1e-8):
            device = next(self.parameters()).device
            n, p = X.shape
            def generate_comMat_hess(p, h):
                combolist = list(itertools.combinations(range(p),2)) 
                comMat=np.zeros([4*int(p*(p-1)/2), p])
                for i in range(len(combolist)):
                    comMat[i*4,combolist[i][0]] = 1
                    comMat[i*4,combolist[i][1]] = 1
                    comMat[i*4+1,combolist[i][0]] = 1
                    comMat[i*4+1,combolist[i][1]] = -1
                    comMat[i*4+2,combolist[i][0]] = -1
                    comMat[i*4+2,combolist[i][1]] = 1
                    comMat[i*4+3,combolist[i][0]] = -1
                    comMat[i*4+3,combolist[i][1]] = -1
                return comMat*h

            def generate_comMat_grad(p, h):
                combolist = list(itertools.combinations(range(p),2)) 
                comMat=np.zeros([4*int(p*(p-1)/2), p])
                for i in range(len(combolist)):
                    comMat[i*4,combolist[i][0]] = 1
                    comMat[i*4+2,combolist[i][0]] = 1
                return comMat*h

            comMat_hess = generate_comMat_hess(p,0.8)
            comMat_grad = generate_comMat_grad(p, 0.8)

            ## cal interaction strength for one sample.
            inter_uncal_arr = np.zeros(int(p*(p-1)/2))
            inter_cal_arr = np.zeros(int(p*(p-1)/2))
            combolist = list(itertools.combinations(range(p),2)) 
            for i in tqdm(range(n)):
                X_dup_hess = np.tile(X[i,:], (4*int(p*(p-1)/2), 1))+comMat_hess
                cc_hess = self(torch.Tensor(X_dup_hess).to(device)).cpu().detach() \
                    .numpy().reshape(int(p*(p-1)/2),4)
                inter_uncal_temp = np.abs(cc_hess[:,0]-cc_hess[:,1]-cc_hess[:,2]+cc_hess[:,3])
                inter_uncal_arr += inter_uncal_temp
                X_dup_grad = np.tile(X[i,:], (4*int(p*(p-1)/2), 1))+comMat_grad
                cc_grad = self(torch.Tensor(X_dup_grad).to(device)).cpu().detach() \
                    .numpy().reshape(int(p*(p-1)/2),4)
                denominator_temp = np.sqrt(np.abs((cc_grad[:, 0] - cc_grad[:, 1])*
                    (cc_grad[:, 2] - cc_grad[:, 3]))) + eps
                inter_cal_temp = inter_uncal_temp.copy() / denominator_temp
                # (np.sqrt(np.abs((cc[:,0]-cc[:,1])*(cc[:,2]+cc[:,3])))+eps)
                inter_cal_arr += inter_cal_temp
            inter_uncal_arr /= n
            inter_cal_arr /= n
            inter_uncal_mat=np.zeros([p,p])
            inter_cal_mat=np.zeros([p,p])
            combolist = list(itertools.combinations(range(p),2))
            for i in range(len(combolist)):
                inter_uncal_mat[combolist[i][0],combolist[i][1]] = inter_uncal_arr[i]
                inter_cal_mat[combolist[i][0],combolist[i][1]] = inter_cal_arr[i]

            interaction_uncal_ranking = []
            interaction_cal_ranking = []
            def inter_func(i, j):
                interaction_uncal_ranking.append(((i, j), inter_uncal_mat[i, j]))
                interaction_cal_ranking.append(((i, j), inter_cal_mat[i, j]))
            # original-original (i!=j)
            for comb in itertools.combinations(np.arange(self.p), 2): inter_func(*comb)
            # knockoff-knockoff (i!=j)
            for comb in itertools.combinations(np.arange(self.p, 2*self.p), 2): inter_func(*comb)
            # original-knockoff combinations
            for i in np.arange(self.p):
                for j in np.arange(self.p, 2*self.p):
                    if (i != j-self.p): inter_func(i, j)
            # knockoff-original combinations
            for i in np.arange(self.p, 2*self.p):
                for j in np.arange(self.p):
                    if (i != j+self.p): inter_func(i, j)
            interaction_uncal_ranking.sort(key=lambda x: x[1], reverse=True)
            interaction_cal_ranking.sort(key=lambda x: x[1], reverse=True)

            if normalize:
                max_score = interaction_uncal_ranking[0][1]
                min_score = interaction_uncal_ranking[-1][1]
                interaction_uncal_ranking = [((i, j), (s-min_score)/(max_score-min_score)) 
                    for (i, j), s in interaction_uncal_ranking]

                max_score = interaction_cal_ranking[0][1]
                min_score = interaction_cal_ranking[-1][1]
                interaction_cal_ranking = [((i, j), (s-min_score)/(max_score-min_score)) 
                    for (i, j), s in interaction_cal_ranking]

            return interaction_uncal_ranking, interaction_cal_ranking

        def shap_feature_interactions(self, X, normalize=False, eps=1e-8):
            interactions = []
            importances = []
            for i in tqdm(range(X.shape[0])):
                game = MachineLearningGame(PytorchMetaGame(self, X), data_index=i)
                shapiq_sii = SHAPIQEstimator(interaction_type="SII", N=set(range(self.p*2)), 
                    order=2,  top_order=False)
                sii_scores = shapiq_sii.compute_interactions_from_budget(
                    game=game.set_call, budget=2**15, show_pbar=True,
                    pairing=True, sampling_kernel="ksh", only_sampling=False, 
                    only_expicit=False, stratification=True)
                importances.append(sii_scores[1])
                interactions.append(sii_scores[2])
            interactions = np.stack(interactions)
            importances = np.stack(importances)
            
            denominator = np.sqrt(np.abs(importances[..., None] * importances[:, None, :]))
            interactions_cal = interactions.copy() / (denominator+eps)

            interactions = np.abs(interactions.mean(0))
            interactions_cal = np.abs(interactions_cal.mean(0))

            interaction_uncal_ranking = []
            interaction_cal_ranking = []
            def inter_func(i, j):
                interaction_uncal_ranking.append(((i, j), interactions[i, j]))
                interaction_cal_ranking.append(((i, j), interactions_cal[i, j]))
            # original-original (i!=j)
            for comb in itertools.combinations(np.arange(self.p), 2): inter_func(*comb)
            # knockoff-knockoff (i!=j)
            for comb in itertools.combinations(np.arange(self.p, 2*self.p), 2): inter_func(*comb)
            # original-knockoff combinations
            for i in np.arange(self.p):
                for j in np.arange(self.p, 2*self.p):
                    if (i != j-self.p): inter_func(i, j)
            # knockoff-original combinations
            for i in np.arange(self.p, 2*self.p):
                for j in np.arange(self.p):
                    if (i != j+self.p): inter_func(i, j)
            interaction_uncal_ranking.sort(key=lambda x: x[1], reverse=True)
            interaction_cal_ranking.sort(key=lambda x: x[1], reverse=True)

            if normalize:
                max_score = interaction_uncal_ranking[0][1]
                min_score = interaction_uncal_ranking[-1][1]
                interaction_uncal_ranking = [((i, j), (s-min_score)/(max_score-min_score)) 
                    for (i, j), s in interaction_uncal_ranking]

                max_score = interaction_cal_ranking[0][1]
                min_score = interaction_cal_ranking[-1][1]
                interaction_cal_ranking = [((i, j), (s-min_score)/(max_score-min_score)) 
                    for (i, j), s in interaction_cal_ranking]

            return interaction_uncal_ranking, interaction_cal_ranking

