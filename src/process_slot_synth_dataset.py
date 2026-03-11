from glob import glob
import json
import os
from utils import create_hash
from tqdm import tqdm
from schema import generate_schema_from_json
import jsonschema
def validate_data(schema, metadata):
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": schema,
        "additionalProperties": False 
    }
    try:
        jsonschema.validate(instance=metadata, schema=schema)
        return True
    except Exception as e:
        return False
        
def remove_spaces_from_keys(obj):
    if isinstance(obj, dict):
        return {
            k.replace(" ", "").replace("-", ""): remove_spaces_from_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [remove_spaces_from_keys(i) for i in obj]
    else:
        return obj

def get_triplets(data, type="slot"):
    if type == "slot":
        output = data["intermediate_output"]
        while isinstance(output, list):
            output = output[0]
        
        schema = output["schema"]
        metadata = output["output"]
        text = data["output"]
    elif type == "slot-reversed":
        schema = data["output"]["schema"]
        metadata = data["output"]["output"]
        text = data["input"]
    schema = remove_spaces_from_keys(schema)
    metadata = remove_spaces_from_keys(metadata)
    return schema, metadata, text
files = []
files += glob("/ibex/ai/home/alyafez/sintetis/.cache/sint-output/18c18daf37/**.jsonl")
files += glob("/ibex/ai/home/alyafez/sintetis/.cache/sint-output/fcc8fa196f/**.jsonl")
files += glob("/ibex/ai/home/alyafez/sintetis/.cache/sint-output/3f35b74824/**.jsonl")
files += glob("/ibex/ai/home/alyafez/sintetis/.cache/sint-output/b6d08ed40a/**.jsonl")
files += glob("/ibex/ai/home/alyafez/sintetis/.cache/sint-output/8a66e47c5b/**.jsonl")
output_path = "static/synth_slot_reversed_slot_with_papersv2"
os.makedirs(output_path, exist_ok=True)
total_items_created = 0
for file in tqdm(files):
    data = open(file, "r").readlines()
    print(len(data))
    for line in data:
        data = json.loads(line)
        if data["output"] is None:
            continue
        slot_type = "slot-reversed"

        if "8a66e47c5b" in file:
            slot_type = "slot"

        try:
            schema, metadata, text = get_triplets(data, type = slot_type)
        except Exception as e:
            print(e)
            continue
        # if not validate_data(schema, metadata):
        #     continue
        # try:
        #     schema_code = generate_schema_from_json(schema, 'CustomSchema')
        #     exec(schema_code)
        # except Exception as e:
        #     # print(schema)
        #     # print(e)
        #     # raise
        #     continue
        if "3f35b74824" in file or "b6d08ed40a" in file:
            link = f"papers_{data['model']}_{slot_type}_{data['id']}"
        else:
            link = f"fineweb_{data['model']}_{slot_type}_{data['id']}"
        metadata = {
            "metadata": metadata,
            "validation": {},
            "cost": {
                "cost": 0,
                "input_tokens": data["num_input_tokens"],
                "output_tokens": data["num_output_tokens"],
            },
            "schema": schema,
            "config": {
                "model_name": data["model"],
                "few_shot": 0,
                "link": link,
                "schema_name": None,
                "context": "all",
                "format": "text",
                "max_model_len": 32768,
                "max_output_len": 2048,
                "browse_web": False,
                "backend": "vllm"
            },
            "error": None
        }
        folder_name = create_hash(metadata["config"]["link"])
        os.makedirs(f"{output_path}/{folder_name}", exist_ok=True)
        with open(f"{output_path}/{folder_name}/{data['id']}.json", "w") as f:
            json.dump(metadata, f)
        
        os.makedirs(f"static/papers/{folder_name}", exist_ok=True)
        with open(f"static/papers/{folder_name}/paper_text.txt", "w") as f:
            f.write(text)
        total_items_created += 1
print("Created ",  total_items_created)
        