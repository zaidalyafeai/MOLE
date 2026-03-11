from search import get_metadata
from rich import print
import json

def extract(text, model_name, schema_name = None, backend = "openrouter", max_model_len = 8192, max_output_len = 2084, schema = None, version = "3.0"):
    message, metadata, cost, error = get_metadata(
        text, model_name, schema_name=schema_name, backend = backend, log = False, max_model_len = max_model_len, max_output_len = max_output_len, schema_json = schema, version = version
    )
    return metadata

def example1(model_name):
    schema = {
            "Name": {
                "type": "string",
                "description": "The name of the model"
            },
            "License": {
                "type": "string",
                "description": "The license of the model"
            },
            "Release_Date": {
                "type": "string",
                "description": "The release date of the model"
            },
            "Architecture": {
                "type": "string",
                "description": "The architecture of the model",
                "enum": ["transformer", "rnn", "cnn", "other"]
            },
            "Versions": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "The versions of the model"
            },
            "Models": {
                "type": "array",
                "items": {
                    "type": "object",   
                    "properties": {
                        "Name": {
                            "type": "string"
                        },
                        "Version": {
                            "type": "number"
                        },
                        "Num_Parameters": {
                            "type": "number"
                        },
                        "Unit": {
                            "type": "string",
                            "enum": [
                                "Million",
                                "Billion",
                                "Trillion"
                            ]
                        }
                    }
                }
            }
    }
    text = """
    Mextract 1.0 are a series of models with sizes: (0.5B, 1.5B, 3B) that can extract metadata from any text. The model has been evaluated on the MOLE
    benchmark and achieved impressive results. The arhitecture is based on Transformer model. The model is released
    under the Apache 2.0 license. The model can process up to 8192 tokens. The model was released in 2025 from KAUST.
    """
    metadata = extract(
        text, model_name, backend = "transformers", schema = schema, version = "3.0",  
    )
    print(metadata)

def example2(model_name, backend="transformers"):
    paper_text = open("static/papers/bbe27eb4/paper_text.txt", "r").read()
    json_schema = json.load(open("src/example_json.json"))
    metadata = extract(
        paper_text, model_name, backend = backend, schema = json_schema, version = "3.0",  
    )
    print(metadata)

model_name = "Qwen2.5-0.5B-Instruct-Qwen2.5-72B-Instruct-sft-merged-r_8_alpha_16-evaluated_dataset_11k-without-object-wrapper"
example2(model_name, backend="transformers")
example2("moonshotai/kimi-k2", backend="openrouter")