# FusedGW Entity Alignment

This is a Python implementation of 

> [A Fused Gromov-Wasserstein Framework for Unsupervised Knowledge Graph Entity Alignment](arxiv)
>
> ***Jianheng Tang**, Kangfei Zhao, Jia Li*
>
> ACL 2023 (Findings) 

Dependencies
--------------------------------
- python 3.10.6
- pytorch 1.13.0
- SentenceTransformer 2.2.2
- argparse 1.1
- dgl 0.9.1


How to run
--------------------------------
All datasets and pretrained embeddings used in the paper are on [google drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing). Download and unzip all files in the `data` folder.

Run `bash run.sh` to reproduce all the experimental results in our paper.