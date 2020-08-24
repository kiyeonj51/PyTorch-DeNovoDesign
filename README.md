# De novo design
The codes below are confirmed for normal operation with AWS Deep Learning AMI (Ubuntu 18.04) Version 32.0 under the pytorch_p36 conda environment.
All codes are downloaded from github ( [REINVENT](https://github.com/MolecularAI/Reinvent), [GENTRL](https://github.com/insilicomedicine/GENTRL), [MOSES](https://github.com/molecularsets/moses) ). After activating pytorch environment, you have to install rdkit package 
```bash
conda install -c conda-forge rdkit 
```
### [REINVENT](https://github.com/MolecularAI/Reinvent)
1. Pre-train <br/> 
Run train_prior.py
2. RL-train <br/> 
Run main.py --scoring-function tanimoto --num-steps 1000 <br/>
(we also can use --scoring-function activity_model, but we have to train classifier for target activity model in advance)

### [GENTRL](https://github.com/insilicomedicine/GENTRL)
1. Pre-train <br/>
Run pretrain.py
2. RL-train <br/>
Run rltrain.py
3. sampling <br/>
Run sampling.py

### [MOSES](https://github.com/molecularsets/moses)
We use part of this code for the moleclue metrics.
