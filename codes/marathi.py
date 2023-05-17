from datasets import load_dataset, load_from_disk
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

eli5 = load_from_disk("marathi_dataset_20k")
df = pd.DataFrame(eli5)
#df2 = df.iloc[0:300]
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mr")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mr")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

paras = []
step = 100
for i in range(0, 20000, step):
    start = time.time()
    # en_text = eli5[i]['src']
    # batch_en.append(en_text)
    batch_en = [eli5[j]['src'] for j in range(i, i + step)]
    batch = tokenizer(batch_en, return_tensors="pt", padding=True).to(device)
    generated_ids = model.generate(**batch, max_length=128)
    para = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(str(i) + "  --- %s seconds ---" % (time.time() - start))
    paras = paras + para

df['paraphrase'] = paras
df.to_csv('marathi_paraphrase.csv', index=False)