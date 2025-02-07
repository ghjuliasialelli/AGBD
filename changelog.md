# Changelog

All notable changes to the dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### 07.02.2024
[This commit](https://github.com/ghjuliasialelli/AGBD/commit/c6bc127dde7dcf9eb11285c8d87e0f55bd4a9829) implements changes that reflect the updated code, when the models were re-trained following the fixing of the latitude and longitude bug. Notably:
- in `dataset.py`, we introduce new encoding strategies for the data (`cat2vec`/`onehot`/`dist`), as well as additional features (`aspect`/`slope`)
- in `inference.py`, we introduce the ability to run inference with an ensemble of models, whereas we had only made code available for running inference on a single model.
- in `models.py`, we corrected the `get_layers()` function, to enforce having UNet models around 10M parameters, regardless of the input `patch_size`.

### 13.11.2024
[This commit](https://github.com/ghjuliasialelli/AGBD/commit/dab81b106fbdb65ff85869c897d2a42802ef0cb6) fixed a bug in `Models/dataset.py` in the computation of the latitude and longitude. 
