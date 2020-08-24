# import gentrl
import os
import torch
from gentrl.model import GENTRL, TrainStats
from gentrl.encoder import RNNEncoder
from gentrl.decoder import DilConvDecoder

cuda = True if torch.cuda.is_available() else False

# torch.cuda.set_device(0)

enc = RNNEncoder(latent_size=50)
dec = DilConvDecoder(latent_input_size=50)
model = GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)
if cuda:
    model.cuda()

model.load('saved_gentrl/')
if cuda:
    model.cuda()

from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol

from moses.utils import disable_rdkit_log
disable_rdkit_log()


def get_num_rings_6(mol):
    r = mol.GetRingInfo()
    return len([x for x in r.AtomRings() if len(x) > 6])


def penalized_logP(mol_or_smiles, masked=False, default=-5):
    mol = get_mol(mol_or_smiles)
    if mol is None:
        return default
    reward = logP(mol) - SA(mol) - get_num_rings_6(mol)
    if masked and not mol_passes_filters(mol):
        return default
    return reward


model.train_as_rl(penalized_logP)

os.makedirs('./saved_gentrl_after_rl', exist_ok=True)
# ! mkdir -p saved_gentrl_after_rl

model.save('./saved_gentrl_after_rl/')