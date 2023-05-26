# web-document-discourse-parsing

This is the repository of web document discourse parsing.

Our code and data will be coming soon.

# requirements
```
 conda create --name [environment_name] python==3.9

 pip install -r pip_requirements.txt

 conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

# train and evaluate models
After set up the requirements, you can train and evaluate the models by:

 ```python train.py [model_name] --[training_options]```
 
where [model_name] can be one of the following:
- `baseline` for NodeBased model in the paper, a naive implementation for the task
- `deepseq` for DeepSeq model [Shi and Huang, 2019](https://aaai.org/papers/07007-a-deep-sequential-model-for-discourse-parsing-on-multi-party-dialogues/)
- `putorskip` for Put-or-Skip model [Cao et al., 2022](https://link.springer.com/article/10.1007/s11390-021-1076-7)
- `ssa` for SSAGNN model  [Wang et al., 2021](https://www.ijcai.org/proceedings/2021/543)
- `damt` for DAMT model in the paper [Fan et al., 2022](https://aclanthology.org/2022.coling-1.76/)

# experimental results


# data
We would like to distribute the data by applying, you can connect me at liupeilin2020@iscas.ac.cn to get the annotated data.

After getting the data, you can unzip the compressed file into `data` folder, there should be 3 splits of data in 3 folders, `data/train`, `data/dev` and `data/test`, respectively.

# Citation
Please cite:
