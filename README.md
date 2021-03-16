ROSITA
========
This repository contains implementation of the paper "ROSITA: Refined BERT cOmpreSsion with InTegrAted techniques
"

Requirements
========
Python3 <br />
torch=1.6.0 <br />
tqdm <br />
boto3 <br />

Data Preparation
========
Step1: Download the [GLUE data](https://gluebenchmark.com/tasks) by running 
```
python download_glue_data.py --data_dir data --tasks all
```
The extracted .tsv files for CoLA and SST-2 are already given in the `dat` folder.

Step2: Conduct data augmentation by running:
