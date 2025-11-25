from pacmap import PaCMAP
from umap import UMAP
import numpy as np
import torch

def reduce(vectors, out_dim, mode='pacmap'):
    if mode == 'pacmap':
        result = PaCMAP(n_components=out_dim, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
        .fit_transform(vectors.numpy())
    if mode == 'umap':
        result = UMAP(n_neighbors=15,min_dist=0.1,metric="euclidean",
        n_components=out_dim,random_state=None,verbose=True).fit_transform(vectors.numpy())
    return result