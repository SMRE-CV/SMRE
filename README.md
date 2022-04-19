code for "SUPPORT-SET BASED MULTI-MODAL REPRESENTATION ENHANCEMENT FOR VIDEO CAPTIONING"(ICME2022)
![image](https://user-images.githubusercontent.com/96560976/164009870-d565cda2-a750-4898-82b1-ad648949226a.png)

# dependencies

torch		    1.8.1

torchtext		0.9.1  

torchvision	0.9.1

python      3.6.9

# Data
**You need to download the following files to reproduce the experiments**

caption-eval & features & text files:

LINK：https://pan.baidu.com/s/13SQdmq0iDHJA2-bcwcDk2A 

PASSWORD：icme

# checkpoint:

***************MSVD**********************

Bleu_1: 84.545195

Bleu_2: 74.130214

Bleu_3: 64.860649

Bleu_4: 55.485736

METEOR: 35.952250

ROUGE_L: 73.029840

CIDEr: 96.142574

*****************************************

LINK：https://pan.baidu.com/s/13kJ32N-C4pkIE-Dlst05zA 

PASSWORD：icme

# Train
You can train the model by running the following command:
```
sh train.sh
```
# Evaluation
```
python evaluate.py --dataset=msvd --model=RMN \
 --result_dir=results/xxxx --attention=soft \
 --hidden_size=1024 --att_size=1024 \
 --test_batch_size=32 --beam_size=5 \
 --eval_metric=CIDEr --topk=18 --max_words=26
```
# Acknowledgement
This repository is partly built based on Ganchao Tan's RMN(https://github.com/tgc1997/RMN) for video captioning.
