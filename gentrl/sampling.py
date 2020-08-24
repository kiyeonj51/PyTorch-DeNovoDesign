# import gentrl
# import torch
import os
from rdkit.Chem import Draw
from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol
from gentrl.model import GENTRL, TrainStats
from gentrl.encoder import RNNEncoder
from gentrl.decoder import DilConvDecoder


# torch.cuda.set_device(0)

enc = RNNEncoder(latent_size=50)
dec = DilConvDecoder(latent_input_size=50)
model = GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)
# model.cuda()

model.load('saved_gentrl_after_rl/')
# model.cuda()

def get_num_rings_6(mol):
    r = mol.GetRingInfo()
    return len([x for x in r.AtomRings() if len(x) > 6])


def penalized_logP(mol_or_smiles, masked=True, default=-5):
    mol = get_mol(mol_or_smiles)
    if mol is None:
        return default
    reward = logP(mol) - SA(mol) - get_num_rings_6(mol)
    if masked and not mol_passes_filters(mol):
        return default
    return reward


generated = []

while len(generated) < 50:
    print(len(generated))
    sampled = model.sample(100)
    sampled_valid = [s for s in sampled if get_mol(s)]

    generated += sampled_valid

os.makedirs('./images', exist_ok=True)
img = Draw.MolsToGridImage([get_mol(s) for s in sampled_valid],
                     legends=[str(penalized_logP(s)) for s in sampled_valid])
img.save('./images/mols.png')

