# Exploring Cross-lingual text style transfer in Multilingual Settings

This repo contains the codebase for the project we worked on for the course COMPSCI 685 in Spring, 2023.

This repository contains the code and resources for our project on few-shot style transfer in low-resource settings. The goal of this project is to develop a model capable of performing targeted style transfer with limited training data. We focus on languages with low resource availability and explore techniques to improve the model's performance in few-shot and zero-shot scenarios. Our main objectives include:

### Project Overview

In this project, we address the challenge of style transfer in low-resource settings by utilizing targeted objectives. We aim to enable the model to perform effective style transfer with limited training data and achieve acceptable performance in few-shot and zero-shot settings. Our main objectives include:

- Developing a model capable of performing few-shot style transfer in low-resource languages.
- Investigating targeted objectives to improve style transfer performance.
- Evaluating the model's performance in few-shot and zero-shot scenarios.
- Exploring the impact of model size and training dataset size on style transfer performance.

![pipeline](https://github.com/Sriharsha-hatwar/multi-lingual-text-style-transfer/blob/main/pipeline.png)

### Repository Structure

The repository is structured as follows:

- `data/`: Contains the datasets used for training and evaluation.
- `models/`: Contains the trained models and checkpoints.
- `notebooks/`: Jupyter notebooks for data preprocessing, training, and evaluation.
- `utils/`: Utility functions and scripts for data processing and model evaluation.
- `results/`: Directory for storing evaluation results and metrics.

For training, we need to run the `objective-1.py`, `objective-1-back-translation.py`, `objective-2.py` and `objective-3.py` inside `implementations` folder. This will generate the pickles file necessary for the experiments. 

### Getting Started

To get started with the project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
    ```
        python 3 
        torch 
        transformers
        numpy 
        typing
        wandb
    ```
3. Prepare the datasets for training and evaluation as described in the data preprocessing notebook.
4. Train the model using the provided training scripts and notebooks.
5. Evaluate the model's performance using the evaluation notebooks.
6. Analyze the results and metrics in the `results/` directory.

### Data
We used the [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) and [mc4](https://huggingface.co/datasets/mc4) datasets for our experiments. The paraphrase dataset needed to train Objective-2 can be generated using `hindi_paraphrase.py` and `marathi_paraphrase.py` inside `implemenations` folder.

### Results and conclusion

Based on our experiments, we have observed that model size and training dataset size significantly impact the performance of few-shot style transfer. We have also identified the limitations of our model in terms of handling multiple style attributes and achieving satisfactory performance in certain scenarios.

In conclusion, this project provides insights into the challenges and potential solutions for few-shot style transfer in low-resource settings. It serves as a foundation for future research in exploring commercially available large language models and adopting advanced prompting strategies for multilingual style transfer tasks.

- `visualizations.ipynb` inside `utils` folder is used to generate the plots for **Loss** and **BLEU Score**. 
- `inference.ipynb` inside `utils` folder is used to do the inferencing experiments. 
- model checkpoints can be found here : https://drive.google.com/drive/folders/1XP77_2lWKlL12JRzHznXFd2VJbs5FaGp?usp=share_link

### Curated Dataset
The dataset we specifically curated to fasciliate multi-lingual, multi-attribute text style transfer can be found in `datasets` folder. It contains files for Hindi, Marathi, Kannada, and Bengali language. There is a file for each language with single attribute (Purity) and multi-attribute (Fomality + Purity). 

### Contributors

- Siddhi Brahmbhatt
- Sriharsha Hatwar
- Hasnain Heickal
- Varad Pimpalkhute (@nightlessbaron)

Feel free to reach out to any of the contributors if you have any questions or feedback about the project.
