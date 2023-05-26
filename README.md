# WebDP: Web-Document-Discourse-Parsing

This is the repository of web document discourse parsing, source code of a paper to be published at findings of ACL 2023. 

In this paper, we proposed a new task named WebDP for facilitating research on discourse parsing of current-days' web-documents on the internet, revealing their free-styled discourse organization and leaveraging their semi-structured information.

Our code and data will be coming soon.

# Requirements
```
 conda create --name [environment_name] python==3.9
 
 conda activate [environment_name]

 pip install -r requirements.txt

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
We would like to distribute the data through applications, you can connect me at liupeilin2020@iscas.ac.cn.

After getting the data, you can unzip the compressed file into `data` folder, there should be 3 splits of data in 3 folders, `data/train`, `data/dev` and `data/test`, respectively.

# Citation
