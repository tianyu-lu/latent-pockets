"""
Compute latent vectors for SMILES strings
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
import pandas as pd
from tqdm.auto import tqdm

from hgraph import HierVAE, PairVocab, common_atom_vocab
from hgraph.hgnn import make_cuda
from hgraph.preprocess import tensorize

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--smiles", required=True)
parser.add_argument("--vocab", required=True)
parser.add_argument("--atom_vocab", default=common_atom_vocab)
parser.add_argument("--save_dir", required=True)
parser.add_argument("--load_model", default=None)
parser.add_argument("--seed", type=int, default=7)

parser.add_argument("--rnn_type", type=str, default="LSTM")
parser.add_argument("--hidden_size", type=int, default=250)
parser.add_argument("--embed_size", type=int, default=250)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--latent_size", type=int, default=32)
parser.add_argument("--depthT", type=int, default=15)
parser.add_argument("--depthG", type=int, default=15)
parser.add_argument("--diterT", type=int, default=1)
parser.add_argument("--diterG", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.0)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--clip_norm", type=float, default=5.0)
parser.add_argument("--step_beta", type=float, default=0.001)
parser.add_argument("--max_beta", type=float, default=1.0)
parser.add_argument("--warmup", type=int, default=10000)
parser.add_argument("--kl_anneal_iter", type=int, default=2000)

parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--anneal_rate", type=float, default=0.9)
parser.add_argument("--anneal_iter", type=int, default=25000)
parser.add_argument("--print_iter", type=int, default=50)
parser.add_argument("--save_iter", type=int, default=5000)

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)

with open(args.vocab) as f:
    vocab = [x.strip("\r\n ").split() for x in f]
args.vocab = PairVocab(vocab, cuda=False)

model = HierVAE(args)


def smiles_to_vec(smiles: str) -> np.ndarray:
    graphs, tensors, orders = tensorize([smiles], vocab=args.vocab)
    tree_tensors, graph_tensors = tensors = make_cuda(tensors)

    with torch.no_grad():
        root_vecs, tree_vecs, _, graph_vecs = model.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = model.rsample(
            root_vecs, model.R_mean, model.R_var, perturb=False
        )

    return root_vecs.detach().cpu().numpy()[0]


df = pd.read_csv(args.smiles)
smiles = df["canonical_smiles"].to_list()
vecs = []
for i, s in tqdm(enumerate(smiles)):
    vec = []
    try:
        vec = list(smiles_to_vec(s))
    except Exception as err:
        logging.warning(f"{s} {err}")
    vecs.append(vec)


df["hgraph"] = vecs
df.to_csv(f"{args.smiles}_embed.csv")
