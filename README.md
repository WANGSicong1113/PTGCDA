# PTGCDA: a PU-learning-driven Transformer and GraphSAGE framework for circRNA-disease association prediction
Copyright (C) 2026 Sicong Wang
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
<img width="832" height="772" alt="image" src="https://github.com/user-attachments/assets/a16a28c8-0e90-4335-b433-751b83d3a7da" />

# Getting Started
# Create environment
conda create -n PTGCDA python=3.10 -y
conda activate PTGCDA
# Install dependencies
pip install numpy==1.25.0
pip install scipy==1.11.1
pip install pandas==1.5.3
pip install openpyxl==3.0.10
pip install scikit-learn==1.2.2
pip install biopython==1.83
pip install gensim==4.3.1
pip install tqdm==4.65.0
pip install matplotlib==3.8.2
pip install jupyterlab==3.6.3
# PyTorch + PyG
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
# Method overview
PTGCDA is a PU-learning-based framework for circRNA–disease association prediction.

The framework integrates:
- Sequence feature extraction using Word2Vec and TextCNN
- Graph representation learning via GraphSAGE
- Cross-modal feature fusion using Transformer Encoder
- Positive–Unlabeled learning to identify reliable negative samples

This design allows PTGCDA to effectively model both sequence-level and network-level information while reducing noise from unlabeled data.
# Input & Output
Input

The model requires the following data:
- circRNA–disease adjacency matrix
- circRNA sequence data
- circRNA similarity matrix (optional)
- disease similarity matrix (optional)

Output

- Prediction scores for circRNA–disease associations
- Ranked candidate associations for downstream validation
# Data
Five benchmark datasets are included:
Dataset1: CircR2Disease
Dataset2: CircR2Cancer
Dataset3: CircRNADisease
Dataset4: Circad
Dataset5: Integrated dataset
# Run main training script
python main.py
