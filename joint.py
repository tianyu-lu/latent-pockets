import argparse
import logging
from pathlib import Path
import random

from Bio.PDB import PDBParser
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import gvp
from embed_pocket import PocketEncoder, pdb_to_json
from hgraph import HierVAE, PairVocab, common_atom_vocab, MolGraph


def to_numpy(tensors):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).long() for x in tree_tensors[:-1]] + [
        tree_tensors[-1]
    ]
    graph_tensors = [make_tensor(x).long() for x in graph_tensors[:-1]] + [
        graph_tensors[-1]
    ]
    return tree_tensors, graph_tensors


parser = argparse.ArgumentParser()
# parser.add_argument("--vocab", type=str, default="/home/tianyu/code/hgraph2graph/data/chembl30/standardized_vocab.txt")
parser.add_argument("--vocab", type=str, default="/home/tianyu/code/hgraph2graph/data/chembl/vocab.txt")
parser.add_argument("--atom_vocab", default=common_atom_vocab)
parser.add_argument("--save_dir", type=str, default="/home/tianyu/code/hgraph2graph/tmp")
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


class JointVAE(nn.Module):
    def __init__(self):
        super(JointVAE, self).__init__()
        self.pocket_encoder = PocketEncoder()
        self.ligand_vae = HierVAE(args)

        # start from pretrained Hgraph weights
        model_state, optimizer_state, total_step, beta = torch.load("/home/tianyu/code/hgraph2graph/ckpt/chembl-pretrained/model.ckpt", map_location=torch.device('cpu'))
        self.ligand_vae.load_state_dict(model_state)

        # freeze weights of ligand generative model
        for param in self.ligand_vae.parameters():
            param.requires_grad = False

    def get_latent(self, pocket):
        """
        Given a pocket, compute its latent representation
        """
        # encode pocket into latent space
        json = [pdb_to_json(pocket)]
        node_counts = [len(entry['seq']) for entry in json]
        sampler = gvp.data.BatchSampler(node_counts, max_nodes=3000)
        dataset = gvp.data.ProteinGraphDataset(json)
        dataloader = DataLoader(dataset, batch_sampler=sampler)

        for batch in dataloader:
            # batch = batch.to(device) # optional
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            root_vecs = self.pocket_encoder(nodes, batch.edge_index, edges)

        return root_vecs

    def forward(self, pocket, ligand, beta=0.1, perturb_z=True):
        """
        pocket: pdb file of pocket residues
        ligand: smiles string of ligand
        """
        root_vecs = self.get_latent(pocket)

        # decode latent into ligand
        # used to compute loss
        graphs, tensors, orders = tensorize([ligand], vocab=args.vocab)
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        root_vecs, kl_div = self.ligand_vae.rsample(root_vecs, self.ligand_vae.R_mean, self.ligand_vae.R_var, perturb_z)
        root_vecs = root_vecs.reshape(1, 32)

        loss, wacc, iacc, tacc, sacc = self.ligand_vae.decoder(
            (root_vecs, root_vecs, root_vecs), graphs, tensors, orders
        )
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc

    def predict(self, pocket, nsamples: int = 10):
        """
        Given a pocket, use its latent representations to generate diverse ligands
        """
        root_vecs = self.get_latent(pocket)

        root_vecs = root_vecs.repeat(nsamples, 1)

        root_vecs, kl_div = self.ligand_vae.rsample(root_vecs, self.ligand_vae.R_mean, self.ligand_vae.R_var, perturb=True)

        smiles_list = self.ligand_vae.decoder.decode(
            (root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150
        )
        return smiles_list


model = JointVAE()
optimizer = torch.optim.Adam(model.parameters())

pdbparser = PDBParser()

df = pd.read_csv("/home/tianyu/code/hgraph2graph/data/chembl30/standardized_data_cleaned.csv")

base_fp = Path("/home/tianyu/code/hgraph2graph/data/structures/pockets_6A")
proteins = [base_fp / f"{chembl_id}_{pdb}_pocket.pdb" for chembl_id, pdb in zip(df["target_chembl_id"], df["pdb"])]
ligands = df["canonical_smiles"].to_list()

losses = []
sampled_smiles = ["original"] + [f"gen_{i+1}" for i in range(10)]

test_fp = Path("/home/tianyu/code/hgraph2graph/data/par2/par2_pocket_6A.pdb")
test_pocket = pdbparser.get_structure(test_fp.stem, test_fp)
test_samples = []

for i, (protein, ligand) in tqdm(enumerate(zip(proteins, ligands))):
    try:
        with torch.no_grad():
            try:
                tensorize([ligand], vocab=args.vocab)
            except:
                logging.warning(ligand)
                continue

        pocket = pdbparser.get_structure(protein.stem, protein)

        optimizer.zero_grad()
        loss, kl, wacc, iacc, tacc, sacc = model(pocket, ligand)
        loss.backward()
        print(loss.item())
        losses.append(loss.item())
        optimizer.step()

        # if loss.item() < 30.0:
        #     try:
        #         samples = model.predict(pocket)
        #         print(f"Protein: {protein.stem}")
        #         print(f"Real ligand: {ligand}")
        #         print(samples)
        #         samples = ",".join(samples)
        #         sampled_smiles.append(f"{ligand},{samples}\n")
        #     except Exception as err:
        #         logging.warning(err)

        # if i % 100 == 0:
        #     try:
        #         samples = model.predict(test_pocket)
        #         test_samples.append(",".join(samples))
        #     except Exception as err:
        #         logging.warning("TEST FAILED")
    except KeyboardInterrupt:
        break


with open("losses.txt", "w") as fp:
    for loss in losses:
        fp.write(f"{loss}\n")

with open("samples.txt", "w") as fp:
    for smiles in sampled_smiles:
        fp.write(smiles)

with open("test_samples.txt", "w") as fp:
    for smiles in test_samples:
        fp.write(smiles)
