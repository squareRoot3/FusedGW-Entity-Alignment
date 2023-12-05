# FusedGW Entity Alignment

This is a Python implementation of 

> [A Fused Gromov-Wasserstein Framework for Unsupervised Knowledge Graph Entity Alignment](https://aclanthology.org/2023.findings-acl.205)
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
All datasets and pretrained embeddings used in the paper are on [google drive](https://drive.google.com/file/d/1QHz6YE7vQBrEuZ1WJIdjQub4O-S_Z2ui/view?usp=sharing). Download and unzip all files in the `data` folder.

Run `bash run.sh` to reproduce all the experimental results in our paper.


```
@inproceedings{FGWEA,
    title = "A Fused {G}romov-{W}asserstein Framework for Unsupervised Knowledge Graph Entity Alignment",
    author = "Tang, Jianheng  and
      Zhao, Kangfei  and
      Li, Jia",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    url = "https://aclanthology.org/2023.findings-acl.205",
    doi = "10.18653/v1/2023.findings-acl.205",
    pages = "3320--3334",
}
```
