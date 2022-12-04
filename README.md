# AE for RHP Design

This code repository is distributed as part of the Nature supplementary software release for *Population-based Heteropolymer Design to Mimic Protein Mixtures in Biological Fluids*. It trains on protein sequences (in their monomer-equivalent form) and provides a meaningful latent space for RHP and protein similarity analysis.


## Data

30,000 membrane protein sequences and 30,000 globular protein sequences with 50% identity threshold were collected from the UniProt database. The sequences are then processed and converted to their monomer-equivalent form. Detailed pre-processing procedures can be found in the paper. Note that the training data is not included in the repo due to Github's file size limit. Please reach out to the corresponding author for data access.

## Package Layout

The directory layout of this repo is adapted from [Pytorch Benchmarks](https://github.com/sparticlesteve/pytorch-benchmarks) repo.

- Configuration files (in YAML format) go in `configs/`. We have included a sample config. It yields reasonable results compared to the published results. The data path in config files must be updated before runing the training script.
- Dataset specifications using PyTorch's Dataset API go into `datasets/`
- Model implementations go into `models/`
- Trainer implementations go into `trainers/`. Trainers inherit from the `base` trainer and are responsible for constructing models as well as training and evaluating them.

Simply run `train_AE.py` to train the model.