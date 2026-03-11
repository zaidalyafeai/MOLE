from datasets import load_from_disk
import json
from rich import print

data = load_from_disk("synth_slot_gemini_flash_newv3").shuffle()
print(data)
for example in data:
    metadata = json.loads(example["metadata"])
    schema = json.loads(example["schema"])
    print(metadata)
    print(schema)
    path = "static/papers/"+example["path"].split("/")[2]+"/paper_text.txt"
    print(path)
    with open(path, "r") as f:
        print('text',f.read())
    break