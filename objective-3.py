import torch, random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, DataCollatorForLanguageModeling
from typing import Dict, List
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names, load_from_disk, load_metric
from torchmetrics import SacreBLEUScore
from tqdm.autonotebook import tqdm
import wandb
import pickle
from datasets import Dataset
import os

# Log in to your W&B account
wandb.login()

metrics_pickle_data = []

class mC4SpanDataset(Dataset):
    def __init__(self, data: Dict[str, List[str]], tokenizer, max_length: int) -> None:
        self.x1: List[str] = data['span1']
        self.x2: List[str] = data['span2']
        self.tokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        return len(self.x1)

    def __getitem__(self, indices: int) -> Dict[str, torch.Tensor]:
        x1 = ['<cls>' + self.x1[index] for index in indices]
        x2 = ['<cls> en ' + self.x2[index] for index in indices] 

        #print(x1, x2, sep = "\n")

        # encoding = self.tokenizer.encode_plus(x1, x2)

        # print(encoding)
        # Tokenize the source and target texts
        x1_tokenized = self.tokenizer(x1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        x2_tokenized = self.tokenizer(x2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        #print(x1_tokenized, x2_tokenized, sep = "\n")

        return {
            'x1_ids': x1_tokenized.input_ids,
            'x2_ids': x2_tokenized.input_ids,
            'x1_masks' : x1_tokenized.attention_mask,
            'x2_masks' : x2_tokenized.attention_mask,
        }

class StyleTransferMT5(nn.Module):
    def __init__(self, model, tokenizer, lx='hi ', max_length = 128):
        super().__init__()
        self.model = model
        self.lx = lx
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

    def forward(self, x1_ids, x1_masks, x2_ids, x2_masks):
        # Make device compatible
        # x1_tokenized = x1_tokenized.to(self.device)
        # x2_tokenized = x2_tokenized.to(self.device)

        x1_ids = x1_ids.to(self.device)
        x2_ids = x2_ids.to(self.device)
        x1_masks = x1_masks.to(self.device)
        x2_masks = x2_masks.to(self.device)

        # Compute style_x1 using MT5 encoder
        style_x1 = self.model.encoder(x1_ids, attention_mask=x1_masks).last_hidden_state[:, 0, :]

        # Feed x2 to MT5 encoder and subtract style_x1 from last hidden state
        x2_encoding = self.model.encoder(x2_ids, attention_mask=x2_masks).last_hidden_state - style_x1.unsqueeze(1)

        #Not sure about this line
        self.model.encoder(x2_ids, attention_mask=x2_masks).last_hidden_state = x2_encoding

        # Generate output sentence x2_en
        x2_en = self.model.generate(
            input_ids=x2_ids,
            attention_mask=x2_ids,
            max_length=self.max_length,
            temperature=1.0,
        )

        # Feed decoded_x2_en to MT5 encoder and add style_x1 to last hidden state
        # lx_encoding = self.tokenizer.encode_plus(f'<cls> {self.lx} ', padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(self.device)
        # x2_en_encoding = torch.cat([lx_encoding, decoded_x2_en], dim=-1)
        # x2_en_encoding = self.model.encoder(decoded_x2_en, attention_mask=x2_tokenized.attention_mask).last_hidden_state + style_x1.unsqueeze(1)

        x2_en_decoded_batch = tokenizer.batch_decode(x2_en) #list of strings
        #print("x2_en_decoded_batch len : ", len(x2_en_decoded_batch))
        x2_en_decoded_token_batch = [] # to take care of logic for prepending tokens

        # Add the langugage token and reverse the process
        for x2_en_decoded in x2_en_decoded_batch:
          
          x2_en_decoded_token = '<cls> ' + self.lx + x2_en_decoded
          x2_en_decoded_token_batch.append(x2_en_decoded_token)
        
        x2_en_decoded_tokenized = self.tokenizer(x2_en_decoded_token_batch, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        # print("x2_en_decoded_tokenized input ids shape: ", x2_en_decoded_tokenized.input_ids.shape)
        # print("style_x1.shape: ", style_x1.shape)
        x2_en_encoding = self.model.encoder(x2_en_decoded_tokenized.input_ids.to(device)).last_hidden_state + style_x1.unsqueeze(1)

        # Generate final output sentence
        decoded_output = self.model.decoder(
            input_ids=x2_ids,
            attention_mask=x2_masks,
            encoder_hidden_states = x2_en_encoding, 
        )

        return x2_ids, decoded_output

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
            model.model.save_pretrained(model_path, from_pt=True)
        self.val_score = epoch_score

def outside_save_checkpoint(model, model_path, epoch_number):
    directory_path = os.path.join(model_path, str(model.lx) + "-" + str(epoch_number))
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    model.model.save_pretrained(directory_path, from_pt=True)

def calculate_validation_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def train(model, batch_size, optimizer, train_loader, num_epochs, loss_criterion):
    model.train()
    batch_size = batch_size
    total_loss = 0
    processed_examples = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tk0 = tqdm(train_loader, total=len(train_loader))
    losses = AverageMeter()
    for index, batch in enumerate(tk0):

        optimizer.zero_grad()
        
        # Get tokenized input spans x1 and x2
        # x1_tokenized = batch['x1_tokenized']
        # x2_tokenized = batch['x2_tokenized']
        x1_ids = batch['x1_ids']
        x2_ids = batch['x2_ids']
        x1_masks = batch['x1_masks']
        x2_masks = batch['x2_masks']

        # Generate output embedding
        x2_ids, x2_reversed = model(x1_ids, x1_masks, x2_ids, x2_masks)

        # Compute Loss
        loss = loss_criterion(model.model.lm_head(x2_reversed.last_hidden_state).view( -1, model.model.lm_head(x2_reversed.last_hidden_state).size(-1) ), x2_ids.view(-1) )
        
        # Update gradients
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        processed_examples += batch_size
        losses.update(loss.item(), x2_ids.shape[0])
        tk0.set_postfix(loss=losses.avg)
        
    avg_loss = total_loss / processed_examples
    
    return avg_loss

def validation(model, batch_size, optimizer, val_loader, num_epochs, loss_criterion):
    model.eval()
    total_loss = 0
    processed_examples = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tk0 = tqdm(val_loader, total=len(val_loader))
    losses = AverageMeter()
    for index, batch in enumerate(tk0):

        with torch.no_grad():
        
          # Get tokenized input spans x1 and x2
          # x1_tokenized = batch['x1_tokenized']
          # x2_tokenized = batch['x2_tokenized']
          x1_ids = batch['x1_ids']
          x2_ids = batch['x2_ids']
          x1_masks = batch['x1_masks']
          x2_masks = batch['x2_masks']

          # Generate output embedding
          x2_ids, x2_reversed = model(x1_ids, x1_masks, x2_ids, x2_masks)

          # Compute Loss
          loss = loss_criterion(model.model.lm_head(x2_reversed.last_hidden_state).view( -1, model.model.lm_head(x2_reversed.last_hidden_state).size(-1) ), x2_ids.view(-1) )
        

        total_loss += loss.item() * batch_size
        processed_examples += batch_size
        losses.update(loss.item(), x2_ids.shape[0])
        tk0.set_postfix(loss=losses.avg)
        
    avg_loss = total_loss / processed_examples
    
    return avg_loss

def run(pretrained_model, device, train_dataset, val_dataset, num_epochs):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #pretrained_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)
    # = pretrained_model
    model = pretrained_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    batch_size = 4
    loss_criterion = nn.CrossEntropyLoss() 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    early_stopping = EarlyStopping(patience = 6, mode = 'min')
    for epoch in range(num_epochs):
        if not early_stopping.early_stop:
            print("Starting epoch :", epoch)
            train_loss = train(model, batch_size, optimizer, train_loader, num_epochs, loss_criterion)
            val_loss = validation(model, batch_size, optimizer, val_loader, num_epochs, loss_criterion)
            val_perplexity = calculate_validation_perplexity(val_loss)
            # Remember that objective two does not require bleu score.
            #val_bleu_score = calculate_validation_bleu(model, val_loader)
            print(f"Epoch {epoch+1}: Train loss - {train_loss:.4f}, Val loss - {val_loss:.4f}")
            early_stopping(val_loss, model, f'./checkpoints/objective_three/model_checkpointed')
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_perplexity" : val_perplexity})
            if early_stopping.early_stop:
                print("early_stop is enforced.")
            # here, we need to store some of the metrics as a pickle file. 
            metrics_pickle_data.append((train_loss, val_loss, val_perplexity))
            outside_save_checkpoint(model, "./checkpoints/per_epoch_objective_three/", epoch)
        else:
            print("No training done for this epoch.")

# Need to change this...
def load_dataset(lang, tokenizer):
    # Please remove the range in the below code.
    if lang == "hindi":
        # Change the below line..
        dataset = load_from_disk("./data/objective_three/hindi_mc4Span_dataset_80k")
        dataset = dataset.shuffle(seed=42).select(range(8000))
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"].shuffle(seed=42)
        val_dataset = dataset["test"].shuffle(seed=42)
        train_dataset_final = mC4SpanDataset(train_dataset, tokenizer, 128)
        val_dataset_final = mC4SpanDataset(val_dataset, tokenizer, 128)
    elif lang =="marathi":
        dataset = load_from_disk("./data/objective_three/marathi_mc4Span_dataset_20k")
        dataset = dataset.shuffle(seed=42).select(range(513))
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"].shuffle(seed=42)
        val_dataset = dataset["test"].shuffle(seed=42)
        train_dataset_final = mC4SpanDataset(train_dataset, tokenizer, 128)
        val_dataset_final = mC4SpanDataset(val_dataset, tokenizer, 128)
    else:
        print("Some issue, exiting.")
        exit(0)
    
    return train_dataset_final, val_dataset_final

def run_for_all_languages(pretrained_model, device, tokenizer, num_epochs):
    languages_used = ["marathi"]
    for index, lang in enumerate(languages_used):
        if lang == "hindi":
            print("Starting Hindi..")
            train_dataset, val_dataset  = load_dataset(lang, tokenizer)
            # Need to check whether the model is available locally with updated parameters...
            model = StyleTransferMT5(pretrained_model, tokenizer, lx = "hi ")
            run(model, device, train_dataset, val_dataset, num_epochs)
            print("Hindi language done..")
        elif lang == "marathi":
            # Read from the checkpoint which stores the best model..
            #checkpoint = torch.load("./checkpoints/objective_one/model_checkpointed.pth")
            print("Starting Marathi..")
            train_dataset, val_dataset  = load_dataset(lang, tokenizer)
            #checkpointed_model = MT5ForConditionalGeneration.from_pretrained("./checkpoints/objective_one/model_checkpointed.pth").to(device)
            pretrained_model = MT5ForConditionalGeneration.from_pretrained('./checkpoints/objective_three/model_checkpointed').to(device)
            checkpointed_model = StyleTransferMT5(pretrained_model, tokenizer, lx = "mr ")
            run(checkpointed_model, device, train_dataset, val_dataset, num_epochs)
        else:
            print("Bailing out..")

    # Now, when metrics_pickle_data is filled, store it as a pickle.
    pickle_data = {"metrics" : metrics_pickle_data}
    with open('./data/pickles/objective_three_metrics.pickle', 'wb') as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
# Load the tokenizer and model
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # replace this with the objective1-backtranslated model.
    pretrained_model = MT5ForConditionalGeneration.from_pretrained("./checkpoints/objective_one_back_translation/model_checkpointed").to(device)
    special_tokens = {'additional_special_tokens': ['<cls>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens)
    pretrained_model.resize_token_embeddings(len(tokenizer))
    num_epochs = 5
    
    wandb.init(
        # Set the project where this run will be logged
        project="objective3_cs685_project_testing", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name="run1", 
        # Track hyperparameters and run metadata
        config={
        "epochs": num_epochs,
        })
    
    run_for_all_languages(pretrained_model, device, tokenizer, num_epochs)