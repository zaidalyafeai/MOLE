from glob import glob
import json 
from utils import convert_to_structured, create_hash
from tqdm import tqdm
import os
import shutil
import signal
from datasets import load_dataset

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def filter_finished(example):
    link = example['url']
    hash = create_hash(link)
    os.makedirs(f"static/papers/{hash}", exist_ok=True)
    path = f"static/papers/{hash}/paper_docling.txt"
    if os.path.exists(path):
        return False
    else:
        return True

def add_url(example):
    if 'url' not in example:
        return{'url': json.loads(open(f"../../MOLE/{example['path']}").read())['config']['link']}
    return example

signal.signal(signal.SIGALRM, handler)

datasets = []
ds = load_dataset('IVUL-KAUST/mextract_dpo', split = 'train')
# ds = ds.filter(lambda x: x['schema_name'] is not None)
# ds = ds.filter(lambda x: 'arxiv' in x['url'])
# ds = ds.shuffle(seed = 42).select(range(0, 5000))

ds = ds.map(add_url)
ds = ds.filter(filter_finished)

# ds.save_to_disk('datasets/mextract_papers_docling_5000')

bs_size = 4
for id in tqdm(range(0, len(ds), bs_size)):

    links = [ds[id + i]['url'] for i in range(bs_size)]
    hashes = [create_hash(link) for link in links]
    for hash in hashes:
        os.makedirs(f"static/papers/{hash}", exist_ok=True)
    paths = [f"static/papers/{hash}/paper_docling.txt" for hash in hashes]
    outputs = convert_to_structured(links)
    for output, path in zip(outputs, paths):
        with open(path, "w") as f:
            f.write(output)
