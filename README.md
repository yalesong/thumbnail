# Yahoo Screen video thumbnail dataset
This repository contains the video thumbnail dataset used in our CIKM 2016 paper **"To Click or Not To Click: Automatic Selection of Beautiful Thumbnails from Videos"** ([arXiv link](https://arxiv.org/abs/1609.01388)).

The code for extracthing video thumbnails is open sourced by Yahoo as part of the Hecate library, available in [this github link](https://github.com/yahoo/hecate).

## Introduction
The dataset consists of 1,118 videos and their associated thumbnails collected from Yahoo Screen. The dataset can be used as a benchmark for automatic video thumbnail extraction techniques.

## The Data
This repository contains four directories. 

The directory `./videos` contains our dataset in the JSON and the TSV formats. Due to the copyright issues, we release public URLs of the videos and their thumbnails. Please contact the leading author for missing videos.

The directory `./labels` contains the label information, i.e., the location of a thumbnail frame in a video. Each file in the directory is named as the unique video ID, and contains SIFTFlow distances between each frame and the ground-truth frame.

The directory `./code` and `./results` contain a MATLAB script and some prediction results for reproducing our results in the paper.

## Baseline Results
You can reproduce the results in our CIKM 2016 paper by using the MATLAB script provided in `code/script_eval.m`. 

In our experiments, we obtained the following mean precision @ k results:

| **Models** | k=1 | K=3 | K=5 |
|:----------:|:---:|:---:|:---:|
|   Random   | 0.0322 | 0.0778 | 0.1127 |
| K-means Centroid | 0.0483 | 0.1091 | 0.1610 |
| K-means Stillness | 0.0555 | 0.1118 | 0.1574 |
| Group LASSO | 0.0349 | 0.0823 | 0.1324 |
| Beauty Rank | 0.125 | 0.0277 | 0.0367 |
| CNN (AlexNet) | 0.0411 | 0.0590 | 0.0689 |
| Ours Supervised | 0.0519 | 0.1190 | 0.1619 |
| **Ours Unsupervised** | **0.0653** | **0.1494** | **0.1896** |



