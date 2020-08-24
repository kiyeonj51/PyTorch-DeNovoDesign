# import gentrl
import os
import torch
import pandas as pd
from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol
from gentrl.model import GENTRL, TrainStats
from gentrl.encoder import RNNEncoder
from gentrl.decoder import DilConvDecoder
from gentrl.dataloader import MolecularDataset

cuda = True if torch.cuda.is_available() else False

# torch.cuda.set_device(0)


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

# ! wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv
# df = pd.read_csv('dataset_v1.csv')

url ="https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
df = pd.read_csv(url)
df = df[:int(len(df)/100)]

df = df[df['SPLIT'] == 'train']
df['plogP'] = df['SMILES'].apply(penalized_logP)
df.to_csv('train_plogp_plogpm.csv', index=None)

enc = RNNEncoder(latent_size=50)
dec = DilConvDecoder(latent_input_size=50)
model = GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)
if cuda:
    model.cuda()

md = MolecularDataset(sources=[
    {'path':'train_plogp_plogpm.csv',
     'smiles': 'SMILES',
     'prob': 1,
     'plogP' : 'plogP',
    }],
    props=['plogP'])

from torch.utils.data import DataLoader
train_loader = DataLoader(md, batch_size=50, shuffle=True, num_workers=0, drop_last=True)

model.train_as_vaelp(train_loader, lr=1e-4)

os.makedirs('./saved_gentrl', exist_ok=True)
# os.mkdir('./saved_gentrl')
# ! mkdir -p saved_gentrl


model.save('./saved_gentrl/')

