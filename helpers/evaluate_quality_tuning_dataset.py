import json 
from glob import glob 
from tqdm import tqdm
import os
from rich import print
import jsonschema
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets
from collections import Counter
import hashlib

output_path = "static/synth_slot_reversed_papersv2"
total_items_created = 0
files = glob(output_path + "/*/*.json")
metrics = {"validation": 0, "list_accuracy": 0, "length": 0}
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
errors = set()
def remove_non_existing_keys(example):
    schema = json.loads(example['schema'])
    metadata = json.loads(example['metadata'])
    try:
        combined_keys = set(schema.keys()) | set(metadata.keys())
    except:
        print(schema)
        print(metadata)
        raise
    for key in combined_keys:
        if key in schema and key not in metadata:
            del schema[key]
        elif key in metadata and key not in schema:
            del metadata[key]
        

    example['schema'] = json.dumps(schema)
    example['metadata'] = json.dumps(metadata)
    return example

def set_default(example):
    schema = json.loads(example['schema'])
    metadata = json.loads(example['metadata'])
    for key in metadata:
        try:
            if metadata[key] is None:
                if key in schema:
                    if schema[key]['type'] == 'string':
                        metadata[key] = ''
                    elif schema[key]['type'] == 'array':
                        metadata[key] = []
        except Exception as e:
            print(e)
            pass
            
    example['metadata'] = json.dumps(metadata)
    return example

def is_list(example):
    schema = json.loads(example['schema'])
    metadata = json.loads(example['metadata'])
    if isinstance(metadata, list) or isinstance(schema, list):
        return False
    return True

def get_empty(example):
    schema = json.loads(example['schema'])
    metadata = json.loads(example['metadata'])
    for key in metadata:
        if metadata[key] == [] and schema[key]['type'] == 'array':
            return True
        if metadata[key] == '' and schema[key]['type'] == 'string':
            return True
    return False

def create_hash(paper_id: str) -> str:
    """Create a hash for a given paper ID."""
    return hashlib.sha256(paper_id.encode()).hexdigest()[:8]

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

def check_description(example):
    schema = json.loads(example["schema"])
    for key in schema:
        if 'description' not in schema[key]:
            return False
    return True

def check_null(example):
    metadata = json.loads(example["metadata"])
    for key in metadata:
        if metadata[key] is None:
            return False
    return True

def is_valid(example):
    metadata = json.loads(example["metadata"])
    schema = json.loads(example["schema"])
    try:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": schema,
            "additionalProperties": False,
            "required": list(schema.keys())
        }
        jsonschema.validate(instance=metadata, schema=schema)
        return True
    except Exception as e:
        # if 'Additional properties are not allowed' in str(e):
        #     return True
        # print(schema)
        # print(metadata)
        errors.add(str(e)[:100])
        return False

def add_num_tokens(example):
    if "docling" in example["source"]:
        paper_text_path = f"static/papers/{create_hash(example['source'])}/paper_docling.txt"
    else:
        paper_text_path = f"static/papers/{create_hash(example['source'])}/paper_text.txt"
    paper_text = open(paper_text_path).read()
    example["num_tokens"] = len(tokenizer.encode(paper_text))
    return example
    
def add_metadata_schema(example):
    path = open(example["path"])
    data = json.load(path)
    example["source"] = data["config"]["link"]
    metadata = data["metadata"]
    example["metadata"] = json.dumps(metadata)
    example["schema"] = json.dumps(data["schema"])
    example["num_attrs"] = len(metadata)
    return example

dataset = Dataset.from_list([{"path": file} for file in files[:100]])
dataset = dataset.map(add_metadata_schema)
dataset = dataset.filter(is_list)
print("dataset after filtering lists")
print(dataset)

print(dataset)
dataset = dataset.map(remove_non_existing_keys)
print("dataset after removing non-existing keys")
print(dataset)

dataset = dataset.map(set_default)
# dataset = dataset.filter(lambda x: "docling" in x["source"])
dataset = dataset.filter(is_valid)
print("dataset after validation")
print(dataset)
print(Counter(errors))
raise


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
# for item in dataset:
#     dataset_attrs.extend(json.loads(item["metadata"]).keys())
# print(len(dataset_attrs))
# print(set(dataset_attrs) & set(attributes))

dataset = dataset.map(fix_array_with_options)
print("dataset after fixing array with options")
print(dataset)


dataset = dataset.filter(check_single_item_lists)
print("dataset after filtering single item lists")
print(dataset)
dataset = dataset.filter(check_style)
print("dataset after filtering style")
print(dataset)

dataset = dataset.filter(check_description)
print("dataset after filtering description")
print(dataset)

dataset = dataset.filter(check_null)
print("dataset after filtering null")
print(dataset)

dataset = dataset.filter(lambda x: 'gemini' in x['source'])
print("dataset after filtering gemini")
print(dataset)

dataset = dataset.filter(lambda x: "fineweb" in x["source"])
print("dataset after filtering fineweb")
print(dataset)

# ds3 = dataset.filter(lambda x: "docling" in x["source"])
# dataset = dataset.filter(lambda x: "fineweb" in x["source"])

# ds1 = dataset.filter(lambda x: get_empty(x))
# print("dataset after with at least one empty")
# print(dataset)
# ds2 = dataset.filter(lambda x: not get_empty(x)).shuffle().select(range(0, 5000-len(ds1)))
# dataset = concatenate_datasets([ds1, ds2, ds3])
# print(dataset)
dataset = dataset.map(add_num_tokens)

dataset = dataset.filter(lambda x: x['num_tokens'] > 50)
print("dataset after filtering num_tokens > 50")
print(dataset)

# dataset = dataset.map(check_options)
# print("dataset after filtering with options")
# print(dataset.filter(lambda x: x["has_options"]))
# print("dataset after filtering with valid options")
# dataset_with_options = dataset.filter(lambda x: x["has_options"]).filter(lambda x: x["valid_options"]).shuffle()
# print(dataset_with_options[0])

# print("dataset from fineweb")
# print(dataset.filter(lambda x: "fineweb" in x["source"]))

# print("dataset from papers")
# print(dataset.filter(lambda x: "papers" in x["source"]))

# dataset.save_to_disk("evaluated_gemini_reversed_only")

