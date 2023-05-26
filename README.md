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
- `baseline` for NodeBased model in the paper
- `deepseq` for DeepSeq model in the paper
- `putorskip` for Put-or-Skip model in the paper
- `ssa` for SSAGNN model in the paper
- `damt` for DAMT model in the paper

# experimental results


# data
We would like to distribute the data by applying, you can connect me at liupeilin2020@iscas.ac.cn to get the annotated data.

After getting the data, you can unzip the compressed file into `data` folder, there should be 3 splits of data in 3 folders, `data/train`, `data/dev` and `data/test`, respectively.

# Citation
Please cite:
