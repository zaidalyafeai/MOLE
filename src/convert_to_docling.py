from glob import glob
import json 
from utils import convert_to_structured, create_hash
from tqdm import tqdm
import os
import shutil
import signal

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

datasets = []
for schema_name in ['ar', 'en', 'jp', 'fr', 'ru', 'multi', 'model', 'tool', 's2orc']:
    datasets += glob(f"evals/{schema_name}/test/**.json")

for file in tqdm(datasets):
    json_data = json.load(open(file))
    link = json_data['Paper_Link']
    hash = create_hash(link)
    path = f"static/papers/{hash}/paper_docling.txt"
    print(path)
    print(link)
    if 'wiki' in link:
        shutil.copy(f"static/papers/{hash}/paper_text.txt", path)
        continue
    
    if os.path.exists(path):
        continue
    try:
        signal.alarm(10)  # timeout after 3 seconds
        structured = convert_to_structured(link)
        signal.alarm(0)  # cancel alarm if finished
    except Exception as e:
        print(e)
        print(link)
        continue
    except TimeoutException:
        print("Function timed out!")
        continue
    
    
    with open(path, "w") as f:
        f.write(structured)
