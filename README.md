ROSITA
========
This repository contains implementation of the paper "ROSITA: Refined BERT cOmpreSsion with InTegrAted techniques
" 

The codes for fine-tuning models (w/o knowledge distillation) are modified from [huggingface/transformers](https://github.com/huggingface/transformers).

The codes for knowledge distillation are modified from [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)

Requirements
========
Python3 <br />
torch=1.6.0 <br />
tqdm <br />
boto3 <br />

Data Preparation
========
Step1: Download the [GLUE data](https://gluebenchmark.com/tasks) by running:
```
python download_glue_data.py --data_dir data --tasks all
```
The extracted .tsv files for CoLA and SST-2 are given in the `data/` folder.

Step2: Download the pre-trained language model BERT (bert-base-uncased) and [GloVe](http://nlp.stanford.edu/data) (glove.42B.300d) embeddings, to `models/bert_pt/` and `glove/` respectively.

Step3: Conduct data augmentation by running:
```
python data_augmentation.py --pretrained_bert_model models/bert_pt \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$
```
