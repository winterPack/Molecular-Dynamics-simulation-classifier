# Nerual Network Classifier for Molecular Dynamics simulation data

This document summarizes the training of a neural network to classify tilt angles with our 3D coarse-grained MD simulation data. 
For an introduction of the dataset, check our paper at DOI:[10.1063/1.4977420](http://aip.scitation.org/doi/abs/10.1063/1.4977420).

## Classification accuracy summary:
- Training set: 93.82%
- Validation set: 90.53%
- Test set: 91.58%

requirements:
- Python 3.5+
- Tensorflow 1.3.0

## To train the model
```bash
python CSP_classification.py
```

## To evaluate the model on a fresh test set
```bash
python evaluation.py
```

Training data is not included in this repository (exceeds the 1G limit). For requests, please contact the paper's correponding author.
