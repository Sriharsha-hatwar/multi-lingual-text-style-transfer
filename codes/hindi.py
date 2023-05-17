from datasets import load_dataset, load_from_disk
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

eli5 = load_from_disk("hindi_dataset_80k")
df = pd.DataFrame(eli5)
# df2 = df.iloc[22600:22700]
# print(df2)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

paras = []
step = 1
for i in range(0, 80000, step):
    start = time.time()
    # en_text = eli5[i]['src']
    # batch_en.append(en_text)
    if len(eli5[i]['src']) > 256: 
        para = [eli5[i]['tgt']] 
        paras = paras + para
        continue
    batch_en = [eli5[j]['src'] for j in range(i, i + step)]
    batch = tokenizer(batch_en, return_tensors="pt", padding=True).to(device)
    generated_ids = model.generate(**batch, max_length=256)
    para = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(str(i) + "  --- %s seconds ---" % (time.time() - start))
    paras = paras + para

df['paraphrase'] = paras
df.to_csv('hindi_paraphrase.csv', index=False)
