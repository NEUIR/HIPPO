# HIPPO
Source code for paper: HIPPO: Enhancing the Table Understanding Capability of Large Language Models through Hybrid-Modal Preference Optimization

## Overview


## Environment Setup

Clone the repository

```
git clone https://github.com/NEUIR/HIPPO.git
cd HIPPO
```

Install Dependencies

```
conda create -n hippo python=3.10
conda activate hippo
pip install -r requirments.txt
```

## Data Preparation
Download the MMTab Image
```
# test
wget https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-eval_table_images_23K.zip
mv MMTab-eval_table_images_23K.zip hippo/
unzip MMTab-eval_table_images_23K.zip

# train
wget https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-instruct_table_images_82K.zip
mv MMTab-instruct_table_images_82K.zip
unzip MMTab-instruct_table_images_82K.zip
```

## Reproduce 

### Train HIPPO
You can download the checkpoint of HIPPO directly from [here](https://huggingface.co/HaolanWang/HIPPO) or go to the ``scripts`` and train the HIPPO model.

For Training, you need to download the model [MiniCPM-V-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) and [data](https://drive.google.com/file/d/15H9IRiL2emMQ8mrMthZWQfCTZbBMKtSC/view?usp=sharing). Then you can go to the ``scripts`` to construct DPO data.

```
cd scripts
bash construct_dpo_data.bash
```

You can also use constructed data directly: [dpo_data](https://drive.google.com/file/d/1KrCD9Zrw8-N7KbRYLwvIn7N-3JUTcn5f/view?usp=sharing).

Then you can train the model.

```
cd scripts
bash train.bash
```



### Inference HIPPO

For Inference, you can go to the ``scripts`` and inference on the HIPPO model: 
```
cd scripts
bash inference.sh
```



### Evaluation

For evaluation, you can use ``src/eval/MMTab_evaluation.ipynb`` to evaluate the performance.
