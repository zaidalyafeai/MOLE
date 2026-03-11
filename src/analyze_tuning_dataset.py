import json
import glob
import argparse
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk
from utils import create_hash
from search import truncate_prompt
import os
from schema import get_schema, Schema
from rich import print


parser = argparse.ArgumentParser(description="Fine-tune model with HuggingFace")
parser.add_argument('--output_model_name', default="qwen2.5-0.5b-instruct-sft", type=str, help="Output directory for the fine-tuned model")
# parser.add_argument('--model_name', default="/hdd/shared_models/Qwen2.5-0.5B-Instruct", type=str, help="Model name to fine-tune")
parser.add_argument('--model_name', default="Qwen2.5-0.5B-Instruct", type=str, help="Model name to fine-tune")
parser.add_argument('--output_dir', default="output", type=str, help="Output directory for the fine-tuned model")
parser.add_argument('--max_model_len', default=8192, type=int, help="Maximum model length")
parser.add_argument('--max_output_len', default=2048, type=int, help="Maximum output length")
parser.add_argument('--distilled_model', default="Qwen2.5-72B-Instruct", type=str, help="Distilled model name")
parser.add_argument('--lora_r', default=8, type=int, help="LoRA rank")
parser.add_argument('--lora_alpha', default=16, type=int, help="LoRA alpha")

args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    trust_remote_code=True,
    padding_side="left",
)

def get_files():
    print('getting synthetic data files')
    all_train_files = load_from_disk("evaluated_dataset")["path"]
    all_train_files = [file for file in all_train_files if args.distilled_model in json.load(open(file))["config"]["model_name"]]
    test_files = []
    valid_files = []
    # valid_files = []  # getting validation from valid files
    for schema_name in ['ar', 'en', 'fr', 'jp', 'ru', 'multi']:
        valid_files += glob.glob(f"evals/{schema_name}/test/*.json")
    
    for schema_name in ['model', 'tool', 's2orc', 'bib']:
        test_files += glob.glob(f"evals/{schema_name}/test/*.json")
        # valid_files += glob.glob(f"evals/{schema_name}/valid/*.json")
    
    # Split training files into train/validation (95%/5%) with reproducible seed
    import random
    rng = random.Random(42)  # Create specific random object with seed
    
    # Shuffle the training files
    shuffled_train_files = all_train_files.copy()
    rng.shuffle(shuffled_train_files)
    
    # Calculate split sizes
    total_train = len(shuffled_train_files)
    val_size = int(0.05 * total_train)
    train_size = total_train - val_size
    
    # Split the files
    train_files = shuffled_train_files
    # valid_files = shuffled_train_files[train_size:]
    
    print(f"Total training files: {total_train}")
    print(f"Train split: {len(train_files)}, Validation split: {len(valid_files)}")
    
    # Old approach - uncomment to use separate validation files instead of splitting training
    # return all_train_files, valid_files, test_files
    
    # return train_files, valid_files, test_files
    return train_files, valid_files, test_files  # Use last 5% of training files as validation

def create_prompts(examples):
    messages = []
    errors = []
    paper_texts = []
    for path in examples['path']:
        data = json.load(open(path))
        if "error" in data:
            errors.append('') if not data['error'] else errors.append(data['error'])
        else:
            errors.append('')
        if "metadata" in data:
            metadata = data['metadata']
            config = data['config'] 
            schema_name = config['schema_name']
            link = config['link']
        else:
            metadata = data.copy()
            del metadata['annotations_from_paper']
            link = metadata['Paper_Link']
            schema_name = path.split('/')[1]
        if schema_name is None:
            schema_json = data["schema"]
        else:
            schema_json = ""
        schema = get_schema(schema_name, schema_json)
        paper_path = f'static/papers/{create_hash(link)}'
        if not os.path.exists(paper_path):
            success, paper_path = download_paper(link, "static/papers/", log=False)
            # raise FileNotFoundError(f"Paper not found at {paper_path}")
        
        paper_text_path = paper_path + "/paper_text.txt"
        paper_text = ''
        if os.path.exists(paper_text_path):
            paper_text = open(paper_text_path, "r").read()
        else:
            try:
                paper_text = extract_paper_text(paper_path, format='pdf_plumber', log=False)
                with open(paper_text_path, "w") as f:
                    f.write(paper_text)
            except Exception as e:
                print(e)
                paper_text = ''
        paper_texts.append(paper_text)
        if schema_name is None:
            prompt, system_prompt = schema.get_prompts_from_schema(paper_text, '', schema_json, version = "3.0")
        else:
            prompt, system_prompt = schema.get_prompts(paper_text, '', version = "3.0")
        prompt = truncate_prompt(prompt, system_prompt, tokenizer, max_model_len=args.max_model_len, max_output_len=args.max_output_len, log=False)
        messages.append([
            {'role': 'system', 'content': system_prompt}, 
            {'role': 'user', 'content': prompt}, 
            {'role': 'assistant', 'content': json.dumps(metadata)}
        ])

    return {"chat": messages, "error": errors, "paper_text": paper_texts}

def by_model(examples):
    output = []
    for path in examples['path']:
        data = json.load(open(path))
        if "config" in data:
            if data['config']['model_name'] == args.distilled_model:
                output.append(True)
            else:
                output.append(False)
        else:
            output.append(True)
    return output

def prepare_dataset(files):
    dataset = Dataset.from_list([{"path": file} for file in files])
    print("num examples: ", len(dataset))
    dataset = dataset.map(create_prompts, batched=True, batch_size=100, num_proc=16)
    print("num examples after creating prompts: ", len(dataset))
    dataset = dataset.filter(lambda x: not bool(x["error"]))
    print("num examples after filtering errors: ", len(dataset))
    
    # Format conversations using apply_chat_template
    def format_chat(example):
        formatted = tokenizer.apply_chat_template(
            example["chat"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": formatted}
    
    dataset = dataset.map(format_chat)
    print("num examples after formatting: ", len(dataset))
    return dataset

def postprocess(output):
    output = output.replace('```json', '').replace('```', '').strip()
    return json.loads(output)

def get_gold_metadata(link):
    files = glob.glob("evals/**/**/*.json")
    for file in files:
        schema_name = file.split('/')[1]
        metadata = json.load(open(file))
        if metadata['Paper_Link'] == link:
            return json.load(open(file)), schema_name
    return None


train_files, valid_files, test_files = get_files()

print(f'train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}')

dataset = prepare_dataset(valid_files)
dataset = dataset.shuffle()
schema = dataset[0]['chat'][1]['content'].split("Paper Text")[0].split("Input Schema:")[1].strip()
text = dataset[0]['chat'][1]['content'].split("Paper Text")[1]
output = dataset[0]['chat'][-1]['content']

print("Schema:")
print(json.loads(schema))
print("\nOutput:")
print(json.loads(output))
print("\nText:")
print(text)
print("\nAllText")
print(dataset[0]["text"])
raise
examples = test_dataset
output_examples = []
results =[]
for i, path in enumerate(examples['path']):
        # Extract only the generated part (after input)
    output_example = {}
    gold_data = json.load(open(path))
    
    schema = None
    if "metadata" in gold_data:
        gold_metadata = gold_data['metadata']
        schema_name = gold_data['config']['schema_name']
        if 'schema' in gold_data:
            schema = gold_data['schema']
    else:
        gold_metadata = gold_data.copy()
        schema_name = path.split('/')[1]
    pred_text = json.dumps(gold_metadata)
    print(pred_text)
    if schema_name is None:
        if schema is not None:
            schema = get_schema(schema=schema)
        else:
            raise('error')
    else:
        schema = get_schema(schema_name)

    try:
        metadata = postprocess(pred_text)
    except Exception as e:
        print(e)
        metadata = schema.generate_metadata(method='default').json()
        print("using default schema")
    pred_metadata = schema(metadata=metadata)
    result = pred_metadata.compare_with(
        gold_metadata,
        return_metrics_only=True,
    )
    print(result['f1'])
    results.append(result)
    output_example['metadata'] = pred_metadata.json()
    output_example['result'] = result
    output_examples.append(output_example)
    raise
