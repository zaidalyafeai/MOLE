from datasets import load_from_disk
from schema import Schema
import json
from tqdm import tqdm
import jsonschema
import glob
import random 
from search import get_metadata
from utils import create_hash, read_json
from rich import print
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")

def add_text(example):
    link = json.load(open(example['path']))['config']['link']
    paper_text = open(f'static/papers/{create_hash(link)}/paper_text.txt').read()
    return {'text': paper_text}
def num_tokens(example):
    paper_text = example['text']
    return {"num_tokens": len(tokenizer(paper_text).input_ids)}

ds = load_from_disk('synth_slot_reversed_slot_with_papersv2')
ds = ds.map(add_text)
ds = ds.map(num_tokens)
ds = ds.filter(lambda x: x['num_tokens'] < 10).shuffle()

for example in tqdm(ds):
    data = json.load(open(example['path']))
    config = data['config'] 
    schema_name = config['schema_name']
    link = config['link']

    schema = json.loads(example["schema"])
    metadata = json.loads(example["metadata"])
    gold_metadata = metadata.copy()
    gold_metadata["annotations_from_paper"] = {k: 1 for k,v in metadata.items()}
    sc = Schema(schema = schema)
    paper_path = f'static/papers/{create_hash(link)}'
        
    paper_text = example['text']
    message, metadata, cost, error = get_metadata(
        paper_text, "google/gemini-2.5-flash-lite", backend = "openrouter", schema_json = schema, version = "3.0", log = False
    )
    try:
        pred_metadata = read_json(metadata)
    except Exception as e:
        pred_metadata = sc.generate_metadata(method = 'random')
    try:
        results = sc.evaluate(pred_metadata, gold_metadata)
        if results["f1"] < 0.6:
            pass
    except Exception as e:
        jsonschema.validate(instance=pred_metadata, schema=schema)
        raise
    
    print(f"F1: {results['f1']}")
    
    
    