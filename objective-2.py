import torch, random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, DataCollatorForLanguageModeling
from typing import Dict, List
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names, load_from_disk, load_metric
from torchmetrics import SacreBLEUScore
from tqdm.autonotebook import tqdm
import wandb
import pickle

# Log in to your W&B account
wandb.login()

metrics_pickle_data = []

class SamanantarHindiTranslationDataset(Dataset):
    def __init__(self, data: Dict[str, List[str]], tokenizer: MT5Tokenizer, source_max_length: int, target_max_length: int):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.source_texts = data["src"]
        self.target_texts = data["tgt"]
        self.language_code = "hi "
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, index):
        source_text =  self.language_code + str(self.source_texts[index])
        # Do we need to add in target text? - I dont think so.
        target_text = str(self.target_texts[index])

        # Tokenize the source and target texts
        source_tokenized = self.tokenizer(source_text, padding="max_length", truncation=True, max_length=self.source_max_length, return_tensors="pt")
        target_tokenized = self.tokenizer(target_text, padding="max_length", truncation=True, max_length=self.target_max_length, return_tensors="pt")

        source_ids = source_tokenized.input_ids.squeeze()
        source_mask = source_tokenized.attention_mask.squeeze()
        target_ids = target_tokenized.input_ids.squeeze()
        target_mask = target_tokenized.attention_mask.squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": target_ids, 
            "decoder_attention_mask": target_mask,
            "labels": target_ids,
            "source_text" : source_text,
            "target_text" : target_text
        }

class SamanantarMarathiTranslationDataset(Dataset):
    def __init__(self, data: Dict[str, List[str]], tokenizer: MT5Tokenizer, source_max_length: int, target_max_length: int):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.source_texts = data["src"]
        self.target_texts = data["tgt"]
        self.language_code = "mr "
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, index):
        source_text =  self.language_code + str(self.source_texts[index])
        # Do we need to add in target text? - I dont think so.
        target_text = str(self.target_texts[index])

        # Tokenize the source and target texts
        source_tokenized = self.tokenizer(source_text, padding="max_length", truncation=True, max_length=self.source_max_length, return_tensors="pt")
        target_tokenized = self.tokenizer(target_text, padding="max_length", truncation=True, max_length=self.target_max_length, return_tensors="pt")

        source_ids = source_tokenized.input_ids.squeeze()
        source_mask = source_tokenized.attention_mask.squeeze()
        target_ids = target_tokenized.input_ids.squeeze()
        target_mask = target_tokenized.attention_mask.squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": target_ids, 
            "decoder_attention_mask": target_mask,
            "labels": target_ids,
            "source_text" : source_text,
            "target_text" : target_text
        }

class MT5ParaphraseModel(nn.Module):
    def __init__(self, model, device):
        super(MT5ParaphraseModel, self).__init__()
        self.device = device
        self.pretrainedmodel = model

    def forward(self, batch_x_ids, batch_x_para_ids):
        with torch.no_grad():
            style_x = self.pretrainedmodel.encoder(batch_x_ids).last_hidden_state[:,0, :]
            style_para_x = self.pretrainedmodel.encoder(batch_x_para_ids).last_hidden_state[:, 0, :]
            s_diff = style_para_x - style_x

        batch_x_para_ids = batch_x_para_ids.type(torch.LongTensor).to(self.device)
        s_diff = torch.unsqueeze(s_diff, dim=1).type(torch.LongTensor).to(self.device)
        batch_x_ids = batch_x_ids.type(torch.LongTensor).to(self.device)

        altered_encoder_hidden_states = self.pretrainedmodel.encoder(batch_x_para_ids).last_hidden_state + s_diff
        outputs = self.pretrainedmodel.decoder(input_ids = batch_x_ids, encoder_hidden_states = altered_encoder_hidden_states)

        return outputs

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            #torch.save({'model_state_dict': model.state_dict()}, model_path)
            model.pretrainedmodel.save_pretrained(model_path, from_pt=True)
        self.val_score = epoch_score

def train(model, batch_size, optimizer, train_loader, num_epochs, loss_criterion, device):
    model.train()
    batch_size = batch_size

    total_loss = 0
    processed_examples = 0
    tk0 = tqdm(train_loader, total=len(train_loader))
    losses = AverageMeter()
    for index, batch in enumerate(tk0):
        optimizer.zero_grad()
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)
        batch_label_ids = batch['labels'].to(device)
        model.zero_grad()
        outputs = model(batch_input_ids=batch_input_ids, batch_attention_mask=batch_attention_mask, batch_labels=batch_label_ids)
        loss = outputs.loss
            
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        processed_examples += batch_size
        losses.update(loss.item(), batch_input_ids.shape[0])
        tk0.set_postfix(loss=losses.avg)
    avg_loss = total_loss / processed_examples
    return avg_loss

def validation(model, batch_size, val_loader, loss_criterion, device):
    model.eval()
    #num_examples = len(train_dataset)
    total_loss = 0.0
    processed_examples = 0
    tk0 = tqdm(val_loader, total=len(val_loader))
    losses = AverageMeter()
    for index, batch in enumerate(tk0):
        with torch.no_grad():
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_label_ids = batch['labels'].to(device)
            outputs = model(batch_input_ids=batch_input_ids, batch_attention_mask=batch_attention_mask, batch_labels=batch_label_ids)
            loss = outputs.loss
        total_loss += loss.item() * batch_size
        processed_examples += batch_size
        losses.update(loss.item(), batch_input_ids.size(0))
        tk0.set_postfix(loss=losses.avg)
    avg_loss = total_loss/processed_examples
    return avg_loss

def calculate_validation_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def calculate_validation_bleu(trained_model, val_loader):
    bleu_score_summed = 0.0
    no_of_batchs = 0
    bleu_metric = SacreBLEUScore()
    total_examples = 0
    print("Calculating BLEU Score")
    for batch in val_loader:
        if total_examples < 200 : # Doing validation BLEU for only 200 samples...
            no_of_batchs += 1
            total_examples = total_examples + len(batch['target_text'])
            batch_target = batch['target_text']
            batch_input_ids = batch['input_ids'].to(device)

            batch_list_target = [[item] for item in batch_target] # Need to convert this way for bleu to work.
            #inputs = tokenizer.batch_encode_plus(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = trained_model.pretrainedmodel.generate(input_ids=batch_input_ids, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
            generated_outputs = []
            for i, output in enumerate(outputs):
                generated_outputs.append(tokenizer.decode(outputs[i], skip_special_tokens=True))

            #bleu_metric = bleu_metric.compute(predictions = generated_outputs, references=batch_list_target)
            #bleu_score_summed+=round(bleu_metric["score"], 1)
            bleu_score = bleu_metric(generated_outputs, batch_list_target)
            bleu_score_summed += bleu_score.item()
        else:
            break
    print("BLEU Score calculation done.")
    return bleu_score_summed / no_of_batchs
        
        
def run(pretrained_model, device, train_dataset, val_dataset, num_epochs):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #pretrained_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)
    # = pretrained_model
    model = pretrained_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    batch_size = 32
    loss_criterion = nn.CrossEntropyLoss() 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    early_stopping = EarlyStopping(patience = 6, mode = 'min')
    for epoch in range(num_epochs):
        if not early_stopping.early_stop:
            print("Starting epoch :", epoch)
            train_loss = train(model, batch_size, optimizer, train_loader, num_epochs, loss_criterion, device)
            val_loss = validation(model, batch_size, val_loader, loss_criterion, device)
            val_perplexity = calculate_validation_perplexity(val_loss)
            # Remember that objective two does not require bleu score.
            #val_bleu_score = calculate_validation_bleu(model, val_loader)
            print(f"Epoch {epoch+1}: Train loss - {train_loss:.4f}, Val loss - {val_loss:.4f}")
            early_stopping(val_loss, model, f'./checkpoints/objective_two/model_checkpointed')
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_perplexity" : val_perplexity})
            if early_stopping.early_stop:
                print("early_stop is enforced.")
            # here, we need to store some of the metrics as a pickle file. 
            metrics_pickle_data.append((train_loss, val_loss, val_perplexity))

        else:
            print("No training done for this epoch.")

def load_dataset(lang, tokenizer):
    # Please remove the range in the below code.
    if lang == "hindi":
        dataset = load_from_disk("./data/hindi_dataset_80k")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"].shuffle(seed=42)
        val_dataset = dataset["test"].shuffle(seed=42)
        train_dataset_final = SamanantarHindiTranslationDataset(train_dataset, tokenizer, 128, 128)
        val_dataset_final = SamanantarHindiTranslationDataset(val_dataset, tokenizer, 128, 128)
    elif lang =="marathi":
        dataset = load_from_disk("./data/marathi_dataset_20k")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"].shuffle(seed=42)
        val_dataset = dataset["test"].shuffle(seed=42)
        train_dataset_final = SamanantarMarathiTranslationDataset(train_dataset, tokenizer, 128, 128)
        val_dataset_final = SamanantarMarathiTranslationDataset(val_dataset, tokenizer, 128, 128)
    else:
        print("Some issue, exiting.")
        exit(0)
    
    return train_dataset_final, val_dataset_final

def run_for_all_languages(pretrained_model, device, tokenizer, num_epochs):
    languages_used = ["hindi", "marathi"]
    for index, lang in enumerate(languages_used):
        if index == 0:
            print("Starting Hindi..")
            train_dataset, val_dataset  = load_dataset(lang, tokenizer)
            # Need to check whether the model is available locally with updated parameters...
            model = MT5ParaphraseModel(pretrained_model, device)
            run(model, device, train_dataset, val_dataset, num_epochs)
            print("Hindi language done..")
        else:
            # Read from the checkpoint which stores the best model..
            #checkpoint = torch.load("./checkpoints/objective_one/model_checkpointed.pth")
            print("Starting Marathi..")
            train_dataset, val_dataset  = load_dataset(lang, tokenizer)
            #checkpointed_model = MT5ForConditionalGeneration.from_pretrained("./checkpoints/objective_one/model_checkpointed.pth").to(device)
            pretrained_model = MT5ForConditionalGeneration.from_pretrained('./checkpoints/objective_two/model_checkpointed').to(device)
            checkpointed_model = MT5ParaphraseModel(pretrained_model, device)
            run(checkpointed_model, device, train_dataset, val_dataset, num_epochs)

    # Now, when metrics_pickle_data is filled, store it as a pickle.
    pickle_data = {"metrics" : metrics_pickle_data}
    with open('./data/pickles/objective_two_metrics.pickle', 'wb') as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
# Load the tokenizer and model
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base").to(device)
    num_epochs = 20
    
    wandb.init(
        # Set the project where this run will be logged
        project="objective1_cs685_project", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name="run1", 
        # Track hyperparameters and run metadata
        config={
        "epochs": num_epochs,
        })
    
    run_for_all_languages(pretrained_model, device, tokenizer, num_epochs)