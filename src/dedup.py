from datasets import load_from_disk
from hashlib import md5
from utils import create_hash
import json

def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False

def get_text(example):
    data = json.load(open(example['path']))
    config = data['config'] 
    schema_name = config['schema_name']
    link = config['link']
    paper_path = f'static/papers/{create_hash(link)}'
        
    paper_text_path = paper_path + "/paper_text.txt"
    paper_text = open(paper_text_path, "r").read()
    return {"text": paper_text, 'hash': create_hash(paper_text)}
    
ds = load_from_disk("synth_slot_reversed_slot_with_papersv2")
print(ds)
ds = ds.map(get_text)
uniques = set(ds.unique("hash"))
ds_filter = ds.filter(check_uniques, fn_kwargs={"uniques": uniques})
print(ds_filter)