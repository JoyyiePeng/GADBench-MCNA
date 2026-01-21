# GADBench-MCNA: Multi-hop Common Neighbor Attention for Graph Anomaly Detection

This repository extends [GADBench](https://github.com/squareRoot3/GADBench) with **MoE-Gated Common Neighbor Attention (MCNA)**, a novel auxiliary module that enhances GNN-based graph anomaly detection by capturing collaborative structures among anomalies.

## Overview

In real-world graphs, anomalies usually exhibit **collaborative structure**, manifested as densely connected sub-graphs. Capturing multi-hop relationships is essential to identify and reinforce the connections among these anomalous nodes.

Existing graph anomaly detection (GAD) methods neglect such structures and face two core challenges:
1. Standard recursive neighbor aggregation paradigm causes **over-squashing** of high-order neighbor information, which is vital for detecting collaborative suspicious behaviors
2. Different collaboration patterns depend on specific hop counts, with irrelevant hops introducing substantial noise

### MoE-Gated Common Neighbor Attention (MCNA)

MCNA is an auxiliary parallel branch that can be integrated with any GNN-based GAD backbone to effectively distinguish collaborative structures and enhance detection performance.

**Key Features:**
- **Explicit Multi-hop Encoding**: Explicitly encodes common neighbor information to capture arbitrary k-hop collaborative structures among anomalies
- **MoE-based Hop Selection**: Employs a Mixture-of-Experts (MoE) structure to choose valuable hops in terms of structural information, balancing information and noise from multi-hop neighbors
- **Gated Fusion**: Fuses structural collaborative patterns (captured by MCNA) and attribute-based patterns (captured by backbones) using a gating network

**Performance Improvements:**
- Average improvements of **5.67%** on AUROC
- Average improvements of **7.61%** on AUPRC
- Tested on 10 real-world datasets with 7 different backbones

## Environment Setup

Before you begin, ensure that you have Anaconda or Miniconda installed on your system. This guide assumes that you have a CUDA-enabled GPU.

```shell
# Create and activate a new Conda environment
conda create -n GADBench-MCNA python=3.9
conda activate GADBench-MCNA

# Install PyTorch and DGL with CUDA 12.4 support
# If you use a different CUDA version, please refer to the PyTorch and DGL websites
conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c dglteam/label/cu124 dgl

# Install additional dependencies
pip install -r requirements.txt
```

## Dataset Preparation

GADBench-MCNA utilizes 16 different datasets (10 from original GADBench + 4 Ethereum datasets + 2 heterogeneous variants).

### Download Datasets

Download the datasets from this [google drive link](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing). After downloading, unzip all files into a folder named `datasets` within the GADBench-MCNA directory.

**Note:** An example dataset `reddit` is included and does not require manual downloading.

### Special Datasets

Due to copyright restrictions:
- **DGraph-Fin**: Download from [DGraph-Fin website](https://dgraph.xinye.com/introduction)
- **Elliptic**: Download from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Preprocessing scripts for these datasets can be found in `datasets/preprocess.ipynb`. You can also preprocess your own dataset according to the notebook.

## Benchmarking

### Basic Usage

Benchmark a model with MCNA on the Reddit dataset:

```bash
python benchmark.py --trial 1 --datasets 0 --models GCN
```

Benchmark multiple models on multiple datasets (10 trials):

```bash
python benchmark.py --trial 10 --datasets 0-9 --models GIN-BWGNN --semi_supervised 1
```

### Advanced Options

**Inductive Setting:**
```bash
python benchmark.py --datasets 5,8 --models GAT-GraphSAGE-XGBGraph --inductive 1
```

**Heterogeneous Graphs:**
```bash
python benchmark.py --datasets 10,11 --models RGCN-HGT-CAREGNN-H2FD
```

**Full Benchmark (requires 48GB+ GPU):**
```bash
python benchmark.py --trial 10 --datasets 0-9
```

### Hyperparameter Optimization

Perform random search for optimal hyperparameters:

```bash
# Single model on single dataset
python random_search.py --trial 100 --datasets 0 --models GCN

# All models on all datasets
python random_search.py --trial 100
```

## MCNA Module Integration

The MCNA module is implemented in `model/ncn_modules.py` and can be integrated with various GNN backbones:

- **SpaceGNN**: See `model/spacegnn.py` for integration example
- **DGA-GNN**: See `model/dgagnn.py` for integration example
- **Graph Transformer**: See `model/attention.py` for GT and GAT integration
- **ARC**: See `model/arc_model.py` for integration example

### Key Components

1. **MultiHopCNComputer**: Precomputes common neighbor information for multiple hops
2. **SparseMultiHopMoE**: MoE-based routing to select valuable hops
3. **GatedFusion**: Fuses backbone and MCNA outputs with learnable gates

## Dataset Information

| ID | Name                                                                                                        |    #Nodes |     #Edges | #Dim. | Anomaly | Train | Relation Concept     | Feature Type      |
| -- | ----------------------------------------------------------------------------------------------------------- | --------: | ---------: | ----: | ------: | ----: | -------------------- | ----------------- |
| 0  | [Reddit](https://github.com/pygod-team/data)                                                                   |    10,984 |    168,016 |    64 |   3.3\% |  40\% | Under Same Post      | Text Embedding    |
| 1  | [Weibo](https://github.com/pygod-team/data)                                                                    |     8,405 |    407,963 |   400 |  10.3\% |  40\% | Under Same Hashtag   | Text Embedding    |
| 2  | [Amazon](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudAmazonDataset.html#dgl.data.FraudAmazonDataset) |    11,944 |  4,398,392 |    25 |   9.5\% |  70\% | Review Correlation   | Misc. Information |
| 3  | [YelpChi](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudYelpDataset.html#dgl.data.FraudYelpDataset)    |    45,954 |  3,846,979 |    32 |  14.5\% |  70\% | Reviewer Interaction | Misc. Information |
| 4  | [Tolokers](https://docs.dgl.ai/en/latest/generated/dgl.data.TolokersDataset.html)                              |    11,758 |    519,000 |    10 |  21.8\% |  40\% | Work Collaboration   | Misc. Information |
| 5  | [Questions](https://docs.dgl.ai/en/latest/generated/dgl.data.QuestionsDataset.html)                            |    48,921 |    153,540 |   301 |   3.0\% |  52\% | Question Answering   | Text Embedding    |
| 6  | [T-Finance](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)                                       |    39,357 | 21,222,543 |    10 |   4.6\% |  50\% | Transaction Record   | Misc. Information |
| 7  | [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)                                       |   203,769 |    234,355 |   166 |   9.8\% |  50\% | Payment Flow         | Misc. Information |
| 8  | [DGraph-Fin](https://dgraph.xinye.com/)                                                                        | 3,700,550 |  4,300,999 |    17 |   1.3\% |  70\% | Loan Guarantor       | Misc. Information |
| 9  | [T-Social](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)                                        | 5,781,065 | 73,105,508 |    10 |   3.0\% |  40\% | Social Friendship    | Misc. Information |
| 10 | Amazon (Hetero)                                                                                             |    11,944 |  4,398,392 |    25 |   9.5\% |  70\% | Review Correlation   | Misc. Information |
| 11 | YelpChi (Hetero)                                                                                            |    45,954 |  3,846,979 |    32 |  14.5\% |  70\% | Reviewer Interaction | Misc. Information |
| 12 | alpha_homora                      | 76,813     | 664,285     | 16    | 8.69%         | 80.00% | Ethereum Transactions | Misc. Information |
| 13 | cryptopia_hacker                  | 208,297    | 773,668     | 16    | 3.99%         | 80.00% | Ethereum Transactions | Misc. Information |
| 14 | plus_token_ponzi                  | 34,712     | 91,264      | 16    | 88.68%        | 80.00% | Ethereum Transactions | Misc. Information |
| 15 | upbit_hack                        | 451,725    | 1,683,563   | 16    | 3.75%         | 80.00% | Ethereum Transactions | Misc. Information |

## Citation

If you use this code or find it useful, please cite our paper:

```bibtex
@article{mcna2024,
  title={MoE-Gated Common Neighbor Attention for Graph Anomaly Detection},
  author={},
  journal={},
  year={}
}
```

### Original GADBench Citation

This work is built upon GADBench. Please also cite the original paper:

```bibtex
@inproceedings{tang2023gadbench,
  author = {Tang, Jianheng and Hua, Fengrui and Gao, Ziqi and Zhao, Peilin and Li, Jia},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {29628--29653},
  title = {GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection},
  url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/5eaafd67434a4cfb1cf829722c65f184-Paper-Datasets_and_Benchmarks.pdf},
  volume = {36},
  year = {2023}
}
```

## Acknowledgments

This project is based on [GADBench](https://github.com/squareRoot3/GADBench). We thank the authors for their excellent work and open-source contribution.

## License

This project follows the same license as the original GADBench repository.
