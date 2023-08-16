# WebDP: Web-Document-Discourse-Parsing

This is the repository of web document discourse parsing, source code of a paper to be published at findings of ACL 2023. 

In this paper, we proposed a new task named WebDP for facilitating research on discourse parsing of current-days' web-documents on the internet, revealing their free-styled discourse organization and leaveraging their semi-structured information.

Our code and data will be coming soon.

# Requirements

```
 conda create --name [environment_name] python==3.9
 
 conda activate [environment_name]

 pip install -r pip_requirements.txt

 conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

# Train and Evaluate Models

After set up the requirements, you can train and evaluate the models to reproduce the results in our paper by:

 ```python train.py [model_name] --[training_options]```

where [model_name] can be one of the followings:

- `baseline` for NodeBased model in the paper, a naive implementation for the task
- `deepseq` for DeepSeq model [(Shi and Huang, 2019)](https://aaai.org/papers/07007-a-deep-sequential-model-for-discourse-parsing-on-multi-party-dialogues/)
- `putorskip` for Put-or-Skip model [(Cao et al., 2022)](https://link.springer.com/article/10.1007/s11390-021-1076-7)
- `ssa` for SSAGNN model  [(Wang et al., 2021)](https://www.ijcai.org/proceedings/2021/543)
- `damt` for DAMT model in the paper [(Fan et al., 2022)](https://aclanthology.org/2022.coling-1.76/)


# Data

We would like to distribute the data through applications, you can connect me at githubuser@qq.com to get the annotated data.

After getting the data, you can unzip the compressed file into `data` folder, there should be 3 splits of data in 3 folders, `data/train`, `data/dev` and `data/test`, respectively.

# Citation
```
@inproceedings{liu-etal-2023-webdp,
    title = "{W}eb{DP}: Understanding Discourse Structures in Semi-Structured Web Documents",
    author = "Liu, Peilin  and
      Lin, Hongyu  and
      Liao, Meng  and
      Xiang, Hao  and
      Han, Xianpei  and
      Sun, Le",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.650",
    doi = "10.18653/v1/2023.findings-acl.650",
    pages = "10235--10258",
    abstract = "Web documents have become rich data resources in current era, and understanding their discourse structure will potentially benefit various downstream document processing applications. Unfortunately, current discourse analysis and document intelligence research mostly focus on either discourse structure of plain text or superficial visual structures in document, which cannot accurately describe discourse structure of highly free-styled and semi-structured web documents. To promote discourse studies on web documents, in this paper we introduced a benchmark {--} WebDP, orienting a new task named Web Document Discourse Parsing. Specifically, a web document discourse structure representation schema is proposed by extending classical discourse theories and adding special features to well represent discourse characteristics of web documents. Then, a manually annotated web document dataset {--} WEBDOCS is developed to facilitate the study of this parsing task. We compared current neural models on WEBDOCS and experimental results show that WebDP is feasible but also challenging for current models.",
}
```

