from pathlib import Path

from Bio.PDB import PDBParser, Selection
from Bio.SeqUtils import seq1
import gvp
import gvp.data
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

parser = PDBParser()


def pdb_to_json(structure):
    """
    Returns a dictionary with format specified by
    https://github.com/drorlab/gvp-pytorch#loading-data
    """
    ret = {"name": structure.get_id()}
    seq = []
    coords = []
    for res in Selection.unfold_entities(structure, "R"):
        n, ca, c = list(res["N"].coord), list(res["CA"].coord), list(res["C"].coord)
        seq.append(seq1(res.get_resname()))
        coords.append([n, ca, c])
    ret["seq"] = "".join(seq)
    ret["coords"] = coords
    return ret


class PocketEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(PocketEncoder, self).__init__()

        kwargs = {"node_s": 256, "node_v": 3}

        in_dims = 6 + 20, 3
        node_dims = kwargs["node_s"], kwargs["node_v"]
        edge_dims = 32, 1
        latent = 250
        self.conv0 = gvp.GVPConv(in_dims, node_dims, edge_dims)
        self.layer_norm = gvp.LayerNorm(node_dims)
        self.conv1 = gvp.GVPConvLayer(node_dims, edge_dims)
        self.conv2 = gvp.GVPConvLayer(node_dims, edge_dims)
        self.latent_mean = nn.Linear(kwargs["node_s"], latent)
        # self.var = nn.Linear(kwargs["node_s"], latent)

        self.W_mean = nn.Linear(latent, 32)
        self.W_var = nn.Linear(latent, 32)

    def rsample(self, z_vecs, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = self.W_mean(z_vecs)
        z_log_var = -torch.abs(self.W_var(z_vecs))
        kl_loss = (
            -0.5
            * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))
            / batch_size
        )
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean

        return z_vecs, kl_loss

    def forward(self, nodes, edge_index, edges, batch_idx):
        nodes = self.conv0(nodes, edge_index, edges)
        nodes = self.layer_norm(nodes)
        nodes = self.conv1(nodes, edge_index, edges)
        nodes = self.layer_norm(nodes)
        nodes = self.conv2(nodes, edge_index, edges)
        nodes = self.layer_norm(nodes)

        # take scalar features only
        hid = scatter_mean(nodes[0], batch_idx.type(torch.LongTensor), dim=0)
        
        return self.latent_mean(hid)


gvp_conv = PocketEncoder()

def embed(structures):
    json = [pdb_to_json(s) for s in structures]

    node_counts = [len(entry['seq']) for entry in json]
    sampler = gvp.data.BatchSampler(node_counts, max_nodes=3000)
    dataset = gvp.data.ProteinGraphDataset(json)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    for batch in dataloader:
        # batch = batch.to(device) # optional
        # import ipdb; ipdb.set_trace()
        nodes = (batch.node_s, batch.node_v)
        edges = (batch.edge_s, batch.edge_v)
        # mean, var = gvp_conv(nodes, batch.edge_index, edges)
        mean = gvp_conv(nodes, batch.edge_index, edges, batch.batch)


if __name__ == "__main__":
    fp = Path("/home/tianyu/code/hgraph2graph/data/structures/pockets_6A/CHEMBL205_1zsb_pocket.pdb")
    structure = parser.get_structure(fp.stem, fp)

    pocket_embedding = embed([structure])
