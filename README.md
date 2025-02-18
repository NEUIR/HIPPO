# HIPPO
Source code for paper: HIPPO: Enhancing the Table Understanding Capability of Large Language Models through Hybrid-Modal Preference Optimization



## Installation

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



## Reproduce HIPPO

### Download MMTab Image
```
# test
wget https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-eval_table_images_23K.zip
mv MMTab-eval_table_images_23K.zip hippo/
unzip MMTab-eval_table_images_23K.zip
```

### Inference HIPPO
For Inference, you can go to the ``scripts`` and inference on the HIPPO model: 
```
cd scripts
bash inference.sh
```

### Evaluation

For evaluation, you can use ``src/eval/MMTab_evaluation.ipynb`` to evaluate the performance.