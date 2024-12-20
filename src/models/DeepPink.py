import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        fc1_input_dim = self.compute_fc_input_dim(input_dim)
        self.fc1 = nn.Linear(fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def compute_fc_input_dim(self, input_dim):
        x = torch.randn(1, 1, input_dim)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x.size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        xs = []
        if self.num_unique_categories > 0:
            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns
    
class DeepPINK(torch.nn.Module):
        def __init__(
                self, 
                p,  
                model_type,
                use_Z_weight=True, 
                normalize_Z_weight=False,
                *args,
                **kwargs
            ):
                super(DeepPINK, self).__init__()
                self.p = p
                self.model_type = model_type
                self.use_Z_weight = use_Z_weight
                if self.use_Z_weight:
                    self.Z_weight = nn.Parameter(torch.ones(2 * p))
                else:
                    assert not normalize_Z_weight
                self.normalize_Z_weight = normalize_Z_weight

                if self.model_type == "mlp":
                    # Create MLP layer layers
                    mlp_layers = []
                    hidden_dims = [self.p if use_Z_weight else self.p*2] + kwargs["hidden_dims"] + [1]
                    for i in range(len(hidden_dims) - 1):
                        mlp_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                        if i+1 == len(hidden_dims) - 1: continue
                        mlp_layers.append(torch.nn.ELU())
                    self.mlp = nn.Sequential(*mlp_layers)
                elif self.model_type == "cnn":
                    self.cnn = CNN1D(input_dim=self.p, output_dim=1)
                elif self.model_type == "transformer":
                    self.ft_transformer = FTTransformer(categories = (),      
                                                        num_continuous = p,               
                                                        dim = 32,                          
                                                        dim_out = 1,                        
                                                        depth = 2,                        
                                                        heads = 8,                         
                                                        attn_dropout = 0.1,                 
                                                        ff_dropout = 0.1)


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
                X = X_pink[:, :self.p] + X_pink[:, self.p:]
            if self.model_type == "mlp":
                X = self.mlp(X)
            elif self.model_type == "cnn":
                X = X.unsqueeze(1)
                X = self.cnn(X)
            elif self.model_type == "transformer":
                with sdpa_kernel([SDPBackend.MATH]):
                    X = self.ft_transformer(x_categ=None, x_numer=X)
            return X
        
        def _get_W(self):
            if self.model_type == "mlp":
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
            else:
                raise NotImplementedError("Weights are only implemented for MLP models.")

        def global_feature_importances(self):
            if self.model_type == "mlp":
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
            else:
                raise NotImplementedError("Feature importances are only implemented for MLP models.")

        def get_weights(self):
            if self.model_type == "mlp":
                weights = []
                for name, param in self.mlp.named_parameters():
                    print(name)
                    if "mlp" in name and "weight" in name:
                        weights.append(param.cpu().detach().numpy())
                return weights
            else:
                raise NotImplementedError("Weights are only implemented for MLP models.")

        def global_feature_interactions(self):
            if self.model_type == "mlp":
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

                    for i, j in itertools.product(np.arange(self.p*2), repeat=2):
                        inter_func(i, j)

                    return attributions, interactions
            else:
                raise NotImplementedError("Feature interactions are only implemented for MLP models.")

        def global_3rd_order_interactions(self):
            if self.model_type == "mlp":
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

                    for i, j, k in itertools.product(np.arange(self.p*2), repeat=3):
                        inter_func(i, j, k)
                    
                    return attributions, interactions_2nd, interactions_3rd
            else:
                raise NotImplementedError("3rd order interactions are only implemented for MLP models.")