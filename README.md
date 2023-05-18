# multi-lingual-text-style-transfer
This repo contains the codebase for the project we worked on for the course COMPSCI 685 in Spring, 2023. 

The main pipeline: 
![pipeline](https://github.com/Sriharsha-hatwar/multi-lingual-text-style-transfer/blob/main/pipeline.png)

## Dependencies
- python 3 
- torch 
- transformers
- numpy 
- typing
- wandb

## Data
We used the [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) and [mc4]()https://huggingface.co/datasets/mc4 datasets for our experiments. The paraphrase dataset needed to train Objective-2 can be generated using `hindi_paraphrase.py` and `marathi_paraphrase.py` inside `implemenations` folder.

## Training
For training, we need to run the `objective-1.py`, `objective-1-back-translation.py`, `objective-2.py` and `objective-3.py` inside `implementations` folder. This will generate the pickles file necessary for the experiments. 

## Results and Evaluation
- `visualizations.ipynb` inside `utis` folder is used to generate the plots for **Loss** and **BLEU Score**. 
- `inference.ipynb` inside `utis` folder is used to do the inferencing experiments. 

## Curated Dataset
The dataset we specifically curated to fasciliate multi-lingual, multi-attribute text style transfer can be found in `datasets` folder. It contains files for Hindi, Marathi, Kannada, and Bengali language. There is a file for each language with single attribute (Purity) and multi-attribute (Fomality + Purity). 
