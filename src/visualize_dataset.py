from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from utils import create_hash
import plotext as plt
from rich import print
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")

def add_text(example):
    link = json.load(open(example['path']))['config']['link']
    if 'docling' in example['source']:
        paper_text = open(f'static/papers/{create_hash(link)}/paper_docling.txt').read()
    else:
        paper_text = open(f'static/papers/{create_hash(link)}/paper_text.txt').read()
    return {'text': paper_text}

def stats(example):
    paper_text = example['text']
    return {"num_tokens_text": len(tokenizer(paper_text).input_ids), "num_tokens_schema": len(tokenizer(example['schema']).input_ids), "num_keys": len(json.loads(example['schema']).keys())}

def get_empty(example):
    schema = json.load(open(example['path']))['schema']
    metadata = json.load(open(example['path']))['metadata']
    for key in metadata:
        if metadata[key] == [] and schema[key]['type'] == 'array':
            return True
        if metadata[key] == '' and schema[key]['type'] == 'string':
            return True
            
    return False

ds1 = load_from_disk('evaluated_gemini_with_empty_and_docling')
ds1= ds1.filter(lambda x: 'docling' in x['source'])
ds2= load_from_disk('diverse_subset_2000')

ds = concatenate_datasets([ds1, ds2])
ds.save_to_disk('diverse_subset_2000_with_docling')
raise
# ds = ds.map(add_text).shuffle()
# ds = ds.map(stats)
# ds = ds.filter(get_empty).shuffle()
# print(ds)
print(json.loads(ds[0]['metadata']))
print(json.loads(ds[0]['schema']))
print(ds[0]['text'])


# ds = ds.filter(lambda x: x['num_tokens'] < 800).shuffle()
# plt.hist(ds['num_tokens_text'], bins=3)
# plt.show()

ar_json_data = json.loads(open("schema/ar.json").read())
keys = list(ar_json_data.keys())
for key in keys:
    if key not in ['Name', 'Link', 'Volume', 'Unit', 'Paper_Title', 'Authors', 'Affiliations']:
        del ar_json_data[key]
print(len(tokenizer(json.dumps(ar_json_data)).input_ids))
plt.hist(ds['num_keys'], bins=3)
plt.show()



all_keys = set() 
x, y = [], []
for i,example in enumerate(ds):
    metadata = json.load(open(example['path']))['metadata']
    metadata_keys = list(metadata.keys())
    for key in metadata_keys:
        all_keys.add(key)
    x.append(i)
    y.append(len(all_keys))
plt.plot(x,y)
plt.show()
    
