import random
import numpy as np

class RangedLFMSampler():
    """
    Ranged Latent Factor Model Sampler
    """
    def __init__(self, p, r, low_list, high_list, seed=42):
        """
        p: number of features
        r: number of latent factor features
        low_list: list of lower bounds for each features
        high_list: list of upper bounds for each features
        """
        random.seed(seed)
        np.random.seed(seed)
        self.p, self.r, = p, r
        self.low_list, self.high_list = low_list, high_list
        self.Lambda = np.random.randn(self.r, self.p)

    def sample(self, n): #, A, response_design: Literal["linear", "nonlinear"]):
        """
        n: number of samples
        A: signal amplitude
        """
        F = np.random.randn(n, self.r)
        E = np.random.randn(n, self.p)
        X = F @ self.Lambda + E
        for i in range(self.p):
            Xi = X[:, i]
            Xi_low, Xi_high = self.low_list[i], self.high_list[i]
            lower_idx = np.where(Xi < Xi_low)[0]
            higher_idx = np.where(Xi > Xi_high)[0]
            if len(lower_idx) != 0 or len(higher_idx) != 0:
                # min max normalization
                Xi = (Xi - Xi.min()) * (Xi_high - Xi_low) / (Xi.max() - Xi.min()) + Xi_low
                X[:, i] = Xi

        # veirification
        for i in range(self.p):
            Xi = X[:, i]
            Xi_low, Xi_high = self.low_list[i], self.high_list[i]
            assert Xi.min() >= Xi_low and Xi.max() <= Xi_high

        return X

class UniformSampler():
    def __init__(self, p, low_list, high_list, seed=42):
        """
        p: number of features
        low_list: list of lower bounds for each features
        high_list: list of upper bounds for each features
        """
        random.seed(seed)
        np.random.seed(seed)
        self.p = p
        self.low_list, self.high_list = low_list, high_list

    def sample(self, n):
        """
        n: number of samples
        """
        X = np.random.uniform(low=self.low_list, high=self.high_list, size=(n, self.p))
        # veirification
        for i in range(self.p):
            Xi = X[:, i]
            Xi_low, Xi_high = self.low_list[i], self.high_list[i]
            assert Xi.min() >= Xi_low and Xi.max() <= Xi_high
            
        return X