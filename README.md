# HotFlip Attack and Defense (Adversarial Training) in TensorFlow

*Latest revision： 2022/03/12*

Revised: 

* IMDB-Adversarial-Training.py    
* IMDB-Testing.py    
* helpers.py   

## Introduction   

This code provide implementation of adversarial attacks and training inspired from (Ren et al., 2019, Ebrahimiet al., 2017, i.e., HotFlip). *Have tested*

The repo Also contains some experiments on Particle Swarm Optimization Attack and some others. *Not test*

## Run Locally

Base on GloVec pretrained word embedding (embeds/glove.6B.50d.txt).  

*  Words and embeddings in Attack have covered all the words in GloVe.

run `IMDB-Testing.py` for HotFlip attacks (Word Level).   
run `IMDB-Adversarial-Training.py` for adversarial training.  

* Epoch < 3: Training victim model using ground truth dataset as **Victim Model**.
* Epoch >=3: (Adversarial training for defense) Training victim model using ground truth and adversarial examples  as **Defense Model**. 

Model has Button **multitaskMode** for the attack guided by two or more victim models.

