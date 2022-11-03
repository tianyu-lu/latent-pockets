import argparse
import logging
from pathlib import Path
import random

from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from diversity import similarity_matrix
from embed_pocket import PocketEncoder, pdb_to_json
import gvp
from hgraph import HierVAE, PairVocab, common_atom_vocab, MolGraph
# from vendi_score import vendi


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
parser.add_argument("--vocab", type=str, default="/home/tianyu/code/hgraph2graph/data/chembl30/high_affinity_vocab.txt")
# parser.add_argument("--vocab", type=str, default="/home/tianyu/code/hgraph2graph/data/chembl/vocab.txt")
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
        # model_state, optimizer_state, total_step, beta = torch.load("/home/tianyu/code/hgraph2graph/ckpt/chembl-pretrained/model.ckpt", map_location=torch.device('cpu'))
        # self.ligand_vae.load_state_dict(model_state)

        # freeze weights of ligand generative model
        # for param in self.ligand_vae.parameters():
        #     param.requires_grad = False

    def get_pocket_latent(self, pockets):
        """
        Given a pocket, compute its latent representation
        """
        # encode pocket into latent space
        json = [pdb_to_json(pocket) for pocket in pockets]
        node_counts = [len(entry['seq']) for entry in json]
        sampler = gvp.data.BatchSampler(node_counts, max_nodes=3000)
        dataset = gvp.data.ProteinGraphDataset(json)
        dataloader = DataLoader(dataset, batch_sampler=sampler)

        for batch in dataloader:
            # batch = batch.to(device) # optional
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            root_vecs = self.pocket_encoder(nodes, batch.edge_index, edges, batch.batch)

        return root_vecs

    def forward(self, pockets, ligands, beta=0.1, perturb_pocket=False, perturb_ligand=True):
        """
        pocket: pdb file of pocket residues
        ligand: smiles string of ligand
        """

        pocket_vecs = self.get_pocket_latent(pockets)
        pocket_vecs, pocket_kl = self.pocket_encoder.rsample(pocket_vecs, perturb=perturb_pocket)

        # featurize ligand
        graphs, tensors, orders = tensorize(ligands, vocab=args.vocab)
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        # latent of ligand computed with Hgraph encoder
        true_root_vecs, _, _, _ = self.ligand_vae.encoder(tree_tensors, graph_tensors)

        # TODO: contrastive loss

        # reparameterization
        true_root_vecs, ligand_kl = self.ligand_vae.rsample(true_root_vecs, self.ligand_vae.R_mean, self.ligand_vae.R_var, perturb=perturb_ligand)
        true_root_vecs = true_root_vecs.reshape(len(ligands), 32)

        root_vecs = torch.cat((true_root_vecs, pocket_vecs), dim=-1)

        # decode latent into ligand
        loss, wacc, iacc, tacc, sacc = self.ligand_vae.decoder(
            (root_vecs, root_vecs, root_vecs), graphs, tensors, orders
        )
        return loss + beta * ligand_kl, ligand_kl.item(), wacc, iacc, tacc, sacc

    def predict(self, pockets, nsamples: int = 10):
        """
        Given pockets, use its latent representations to generate diverse ligands
        """
        pocket_vecs = self.get_pocket_latent(pockets)

        normal = torch.randn_like(pocket_vecs)

        root_vecs = torch.cat((normal, pocket_vecs), dim=0)

        root_vecs = root_vecs.repeat(nsamples, 1)

        root_vecs, kl_div = self.ligand_vae.rsample(root_vecs, self.ligand_vae.R_mean, self.ligand_vae.R_var, perturb=True)

        smiles_list = self.ligand_vae.decoder.decode(
            (root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150
        )
        return smiles_list


model = JointVAE()
optimizer = torch.optim.Adam(model.parameters())

pdbparser = PDBParser()

df = pd.read_csv("/home/tianyu/code/hgraph2graph/data/chembl30/standardized_data_high_affinity.csv")

base_fp = Path("/home/tianyu/code/hgraph2graph/data/structures/pockets_6A")
proteins = [base_fp / f"{chembl_id}_{pdb}_pocket.pdb" for chembl_id, pdb in zip(df["target_chembl_id"], df["pdb"])]
ligands = df["canonical_smiles"].to_list()

losses = []
sampled_smiles = ["pdb", "original"] + [f"gen_{i+1}" for i in range(10)]
with open("samples.txt", "w") as fp:
    fp.write(",".join(sampled_smiles) + "\n")
with open("diversity.txt", "w") as fp:
    fp.write("Tanimoto,Vendi Score\n")

test_fp = Path("/home/tianyu/code/hgraph2graph/data/par2/par2_pocket_6A.pdb")
test_pocket = pdbparser.get_structure(test_fp.stem, test_fp)
test_samples = []

examples = list(zip(proteins, ligands))

# shuffle input order
random.shuffle(examples)
n_train = int(0.8 * len(examples))
n_val = int(0.10 * len(examples))
train_examples = examples[:n_train]
val_examples = examples[n_train:n_train+n_val]
test_examples = examples[n_train+n_val:]


def chunk(iterable, n=8):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def passing(proteins, ligands):
    p_proteins, p_ligands = [], []
    for i, ligand in enumerate(ligands):
        with torch.no_grad():
            try:
                tensorize([ligand], vocab=args.vocab)
                if Path(proteins[i]).exists():
                    p_ligands.append(ligand)
                    p_proteins.append(proteins[i])
            except:
                logging.warning(ligand)
                continue
    return p_proteins, p_ligands


for i, batch in tqdm(enumerate(chunk(train_examples))):
    try:
        proteins, ligands = zip(*batch)

        # try:
        p_proteins, p_ligands = passing(proteins, ligands)
        if len(p_ligands) == 0:
            continue

        pockets = [pdbparser.get_structure(p.stem, p) for p in p_proteins]

        optimizer.zero_grad()
        loss, kl, wacc, iacc, tacc, sacc = model(pockets, p_ligands)
        loss.backward()
        print(loss.item() / 1.0)
        with open("train_losses.txt", "a+") as fp:
            fp.write(f"{loss.item() / 1.0}\n")
        optimizer.step()

        if loss.item() < 10.0:
            try:
                samples = model.predict(pockets)
                with open("samples.txt", "a+") as fp:
                    for i, smp in enumerate(chunk(samples, n=10)):
                        samples_txt = ",".join(smp)
                        fp.write(f"{p_proteins[i]},{p_ligands[i]},{samples_txt}\n")

                mat = similarity_matrix(samples)
                tanimoto = np.mean(mat)
                # vend = vendi.score_K(mat)
                vend = 0.0
                with open("diversity.txt", "a+") as fp:
                    fp.write(f"{tanimoto},{vend}\n")
            except Exception as err:
                logging.warning(err)
        if i % 10 == 0:
            random.shuffle(val_examples)
            for j, batch in tqdm(enumerate(chunk(val_examples))):
                proteins, ligands = zip(*batch)
                p_proteins, p_ligands = passing(proteins, ligands)
                if len(p_ligands) != 0:
                    pockets = [pdbparser.get_structure(p.stem, p) for p in p_proteins]
                    with torch.no_grad():
                        loss, kl, wacc, iacc, tacc, sacc = model(pockets, p_ligands)
                    with open("val_losses.txt", "a+") as fp:
                        fp.write(f"{loss.item() / 1.0}\n")
                break
        if i % 100 == 0:
            try:
                samples = model.predict([test_pocket])
                with open("test_samples.txt", "a+") as fp:
                    fp.write(",".join(samples) + "\n")
            except Exception as err:
                with open("test_samples.txt", "a+") as fp:
                    fp.write("TEST FAILED\n")
    except KeyError as err:
        logging.warning(err)
        continue
