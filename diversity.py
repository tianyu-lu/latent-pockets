from rdkit import Chem
import deepchem as dc
import numpy as np
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt
from vendi_score import vendi

featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)

def tanimoto_ecfp(smi1, smi2):
    """
    Compute the Tanimoto coefficient between two SMILES strings using ECFP4 fingerprints
    """
    vec1 = featurizer.featurize([smi1])
    vec2 = featurizer.featurize([smi2])
    return np.sum(np.logical_and(vec1, vec2)) / np.sum(np.logical_or(vec1, vec2))


def similarity_matrix(smiles):
    """
    Given a list of SMILES strings, compute the similarity matrix between all pairs
    """
    mat = np.zeros((len(smiles), len(smiles)))
    for i in range(0, len(smiles)-1):
        for j in range(i+1, len(smiles)):
            sim = tanimoto_ecfp(smiles[i], smiles[j])
            mat[i, j] = sim

    return mat + mat.T + np.eye(len(smiles))


# smiles = ['CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1Cl', 'CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1Cl', 'CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1S(=O)(=O)N1CCCC1', 'CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1S(=O)(=O)N1CCCC1', 'CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1S(=O)(=O)N1CCCC1', 'CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1Cl', 'CC(=O)Nc1ccc(Cl)c(S(=O)(=O)N2CCCC2)c1', 'CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1S(=O)(=O)N1CCCC1', 'CC(=O)Nc1ccc(S(=O)(=O)Nc2ccc(NC(=O)CN3C(=O)NC(=O)C3=O)cc2Cl)cc1', 'CC(=O)Nc1ccc(S(=O)(=O)N2CCCC2)cc1S(=O)(=O)N1CCCC1']
smiles = ['Cc1ccc(Cl)c(Cl)c1Cl', 'COc1ccc(C(=O)Nc2ccc(Cl)c(Cl)c2)cc1', 'COc1ccc(C(=O)Nc2ccc(Cl)c(Cl)c2)cc1Cl', 'COc1ccc(C(=O)Nc2ccc(Cl)c(Cl)c2)cc1Cl', 'Cc1ccc(Cl)c(Cl)c1Cl', 'Cc1ccc(Cl)c(Cl)c1Cl', 'COc1ccc(C(=O)Nc2ccc(Cl)c(Cl)c2)cc1Cl', 'Cc1ccc(Cl)c(Cl)c1Cl', 'Cc1ccc(Cl)c(Cl)c1Cl', 'Oc1ccc(Cl)c(Cl)c1Cl']

mat = similarity_matrix(smiles)

print()
print(f"Vendi = {vendi.score_K(mat)} | Avg = {np.mean(mat)}")

sns.clustermap(mat)
plt.show()
