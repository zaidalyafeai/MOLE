import json 
from glob import glob 
from tqdm import tqdm
import os
from rich import print
import jsonschema
from transformers import AutoTokenizer
from datasets import Dataset
from collections import Counter

output_path = "static/synth_slot_reversed_slot_with_papersv2"
total_items_created = 0
files = glob(output_path + "/*/*.json")
metrics = {"validation": 0, "list_accuracy": 0, "length": 0}
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def check_style(example):
    metadata = json.loads(example["metadata"])
    schema = json.loads(example["schema"])
    if len(set(metadata.keys()) - set(schema.keys())) > 0:
        return False
    return True

def check_single_item_lists(example):
    metadata = json.loads(example["metadata"])
    schema = json.loads(example["schema"])
    for key in schema:
        if "enum" in schema[key]:
            if len(schema[key]["enum"]) == 1:
                return False
        if "items" in schema[key]:
            if "enum" in schema[key]["items"]:
                if len(schema[key]["items"]["enum"]) == 1:
                    return False
    return True

def fix_array_with_options(example):
    example["schema"] = example["schema"].replace('list', 'array')
    metadata = json.loads(example["metadata"])
    schema = json.loads(example["schema"])
    for key in schema:
        try:
            if schema[key]["type"] == "array":
                if "enum" in schema[key]:
                    schema[key]["items"] = {"enum" : schema[key]["enum"]}
                    del schema[key]["enum"]
        except:
            # print(schema)
            pass
                        
    return {"schema": json.dumps(schema)}

def check_options(example):
    metadata = json.loads(example["metadata"])
    schema = json.loads(example["schema"])
    has_options = False
    valid_options = True
    for key in schema:
        if schema[key]["type"] == "array":
            options = []
            if "enum" in schema[key]:
                options.extend(schema[key]["enum"])
            if "items" in schema[key]:
                if "enum" in schema[key]["items"]:
                    options.extend(schema[key]["items"]["enum"])
            if len(options) > 0:
                has_options = True
                # check if the metadata has more list items than options
                if len(metadata[key]) > len(options):
                    valid_options = False
                # check if the metadata returns all options
                try:
                    if len(set(options) - set(metadata[key])) == 0:
                        valid_options = False
                except:
                    print(metadata[key])
                    print(options)
    return {"has_options": has_options, "valid_options": valid_options}

def evaluate_tokenizer(example):
    tokens = tokenizer.encode(example["metadata"])
    return {"num_tokens": len(tokens)}

def is_valid(example):
    metadata = json.loads(example["metadata"])
    schema = json.loads(example["schema"])
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
        # if 'Additional properties are not allowed' in str(e):
        #     return True
        # print(schema)
        # print(metadata)
        # print(e)
        return False
    
def add_metadata_schema(example):
    data = json.load(open(example["path"]))
    metadata = data["metadata"]
    example["metadata"] = json.dumps(metadata)
    example["schema"] = json.dumps(data["schema"])
    example["source"] = data["config"]["link"]
    example["num_attrs"] = len(metadata)
    return example

dataset = Dataset.from_list([{"path": file} for file in files])
dataset = dataset.map(add_metadata_schema)
print("dataset")
print(dataset)

dataset = dataset.filter(is_valid)
print("dataset after validation")
print(dataset)

print(Counter(dataset["num_attrs"]))
attributes = [
    "License",
    "Link",
    "HF_Link",
    "Year",
    "Domain",
    "Form",
    "Collection_Style",
    "Description",
    "Ethical_Risks",
    "Provider",
    "Derived_From",
    "Paper_Title",
    "Paper_Link",
    "Tokenized",
    "Host",
    "Access",
    "Cost",
    "Test_Split",
    "Tasks",
    "Venue_Title",
    "Venue_Type",
    "Venue_Name",
    "Authors",
    "Affiliations",
    "Abstract",
    "Dialect",
    "Language",
    "Script",
]
dataset_attrs = []
for item in dataset:
    dataset_attrs.extend(json.loads(item["metadata"]).keys())
print(len(dataset_attrs))
print(set(dataset_attrs) & set(attributes))

dataset = dataset.map(fix_array_with_options)
print("dataset after fixing array with options")
print(dataset)


dataset = dataset.filter(check_single_item_lists)
print("dataset after filtering single item lists")
print(dataset)
dataset = dataset.filter(check_style)
print("dataset after filtering style")
print(dataset)
# dataset = dataset.map(check_options)
# print("dataset after filtering with options")
# print(dataset.filter(lambda x: x["has_options"]))
# print("dataset after filtering with valid options")
# dataset_with_options = dataset.filter(lambda x: x["has_options"]).filter(lambda x: x["valid_options"]).shuffle()
# print(dataset_with_options[0])

print("dataset from fineweb")
print(dataset.filter(lambda x: "fineweb" in x["source"]))

print("dataset from papers")
print(dataset.filter(lambda x: "papers" in x["source"]))
print(dataset.shuffle()[0])
dataset.save_to_disk("synth_slot_reversed_slot_with_papersv2")

