from pacmap import PaCMAP
from umap import UMAP
import numpy as np
import torch

def reduce(vectors, out_dim, mode='gaussian'):
    '''
    Reduces dimensionality of vector to specified length using specified reduction technique.
    Expects tensor or numpy input.
    '''
    if vectors is not None:
        if torch.is_tensor(vectors) is True:
            if vectors.device == 'cuda':
                vectors = vectors.cpu()
            if vectors.requires_grad is True:
                vectors = vectors.detach()
        if mode == 'pacmap':
            result = PaCMAP(n_components=out_dim, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0).fit_transform(vectors)
        elif mode == 'umap':
            result = UMAP(n_neighbors=15,min_dist=0.1,metric="euclidean", n_components=out_dim,random_state=None,verbose=True).fit_transform(vectors)
        elif mode == 'gaussian':
            gaussian = np.random.randn(vectors.shape[1],out_dim)
            result = vectors @ gaussian
    return result