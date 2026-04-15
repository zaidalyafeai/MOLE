from datasets import load_from_disk, load_dataset, concatenate_datasets
from utils import create_hash
import os
import json

ds = load_dataset('IVUL-KAUST/mextract_dpo', split = 'train')
ds2 = load_from_disk('diverse_subset_2000')

def modify_path(example):
    path = f"../../MOLE/{example['path']}"
    return {'path': path, 'url': json.loads(open(path).read())['config']['link'] }

def filter_dataset(example):
    url = example['url']
    hash = create_hash(url)
    path = f"static/papers/{hash}/paper_docling.txt"
    if os.path.exists(path):
        return True
    else:
        return False

def add_text(example):
    url = example['url']
    hash = create_hash(url)
    path = f"static/papers/{hash}/paper_docling.txt"
    return {"docling_text": open(path).read()}


ds = ds.map(modify_path)
ds = ds.filter(filter_dataset)

ds.save_to_disk('mextract_papers_docling')
ds = concatenate_datasets([ds, ds2])

# remove all columns except path
ds = ds.remove_columns([col for col in ds.column_names if col != 'path'])
ds.save_to_disk('diverse_with_mextract_papers_docling')
print(ds)
# filtered_ds = filtered_ds.map(add_text).shuffle()

# new_column = range(len(filtered_ds))
# filtered_ds = filtered_ds.add_column("id", new_column)
# print(filtered_ds[0])
# path = "/ibex/ai/home/alyafez/sintetis/.cache/sint-input/mextract_papers_docling/"
# os.makedirs(path, exist_ok=True)    
# filtered_ds.to_parquet(f"{path}000_0000.parquet")


