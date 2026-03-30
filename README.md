# PTGCDA
PTGCDA: a PU-learning-driven Transformer and GraphSAGE framework for circRNA-disease association prediction
PTGCDA
License
Copyright (C) 2026 Sicong Wang
This project is intended for academic use only. For commercial use, please contact the author.
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License (GPL v3) or any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details: http://www.gnu.org/licenses/
Type
Package
Overview
PTGCDA is a novel computational framework for predicting circRNA–disease associations (CDAs).
It integrates:
Sequence representation learning (Word2Vec + TextCNN)
Inductive graph learning (GraphSAGE)
Cross-feature attention mechanism
Positive–Unlabeled (PU) learning for reliable negative sampling
The model is designed to address:
Insufficient utilization of circRNA sequence information
High sparsity in heterogeneous biological networks
Noise in negative samples
Files
1. Data
Five benchmark datasets are included:
Dataset1: CircR2Disease
Dataset2: CircR2Cancer
Dataset3: CircRNADisease
Dataset4: Circad
Dataset5: Integrated dataset
2. Code
Core scripts:
similarity1.py → circRNA similarity calculation
association_matrix3.py → adjacency matrix construction
gaussianSimilarity4.py → GIP kernel similarity
descriptor5.py → feature construction
buildTrainset6.py → training dataset generation
mmodels.py → model architecture (GraphSAGE + Attention + PU learning)
fivefold.py → 5-fold cross-validation
Framework
The PTGCDA pipeline consists of three main stages:
Sequence Encoding
k-mer segmentation (k=3)
Word2Vec embedding
TextCNN for feature extraction
Graph Representation Learning
Construction of heterogeneous network
GraphSAGE for node embedding
Association Prediction
Bidirectional attention mechanism
Inner product scoring
Additionally, a two-step PU learning strategy is used to identify reliable negative samples.
Requirements
Python 3.7 (64-bit)
TensorFlow 1.14.0 (GPU recommended)
Keras 2.2.0
Numpy 1.18.0
Gensim 3.8.3
Scikit-learn
Ubuntu 18.04 (recommended)
Usage
Run 5-fold cross-validation:
python fivefold.py
Results
PTGCDA achieves strong performance across multiple datasets:
Superior AUC and AUPR compared to state-of-the-art methods
Robust performance under:
Cold-start scenarios
Highly sparse data conditions
Citation
If you use this work, please cite:
PTGCDA: a PU-learning-driven Transformer and GraphSAGE framework 
for circRNA–disease association prediction
Contact
If you have any questions or suggestions, please contact:
swanggn@connect.ust.hk
GitHub
Code is available at:
https://github.com/WANGSicong1113/PTGCDA
