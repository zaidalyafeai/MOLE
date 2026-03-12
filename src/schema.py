# type: ignore
from pydantic import model_validator
import json
from type_classes import *
from glob import glob

SCHEMA_NAMES = ["ar", "en", "jp", "fr", "ru", "multi", "model", "tool", "s2orc", "bib"]
class Schema:
    def __init__(self, schema_path = None, schema = None, schema_name = None, version = "3.0"):
        self.schema_name = schema_name
        self.version = version

        if schema_name in SCHEMA_NAMES:
            with open(f"schema/{schema_name}.json") as f:
                self.schema = json.load(f)

            self.guidelines = self.process_guidelines(schema_name = schema_name)

            if version == "3.0":
                # print(self.schema)
                self.schema = self.mole_to_slot()
            elif version == "2.0":
                self.schema = self.mole_to_mextract()
            else:
                self.schema = schema
        elif schema_path is not None:
            with open(schema_path) as f:
                self.schema = json.load(f)
        elif schema is not None:
            self.schema = schema
        else:
            raise ValueError("Either schema_name or schema_path must be provided")
        
        self.guidelines = self.process_guidelines(schema_name = schema_name)

    def get_schema_name(self):
        return self.schema_name
    
    def get_eval_datasets(self, split = 'test', path = "evals"):
        datasets = []
        for file in glob(f'{path}/{self.get_schema_name()}/{split}/**.json'):
            data = json.load(open(file))
            datasets.append(data)
        return datasets
    

    def get_attributes(self):
        return [key for key in self.schema.keys()]

    def get_schema(self):    
        return self.schema
    
    def dump_schema(self):
        return json.dumps(self.schema, indent=4)
    
    def get_prompt(self, paper_text, readme = "", metadata = ""):
        if not isinstance(self.schema, str):
            schema = self.dump_schema()
            
        if readme != "":
            prompt = f"""
                    You have the following Metadata: {metadata} extracted from a paper and the following Readme: {readme}
                    Given the following Input schema: {schema}, then update the metadata in the Input schema with the information from the readme.
                    """
        else:  
            prompt = f"""Input Schema: {schema}
                        Paper Text: {paper_text}
                    """
        return prompt
            
    def get_prompts(self, paper_text, readme, metadata):
        prompt = self.get_prompt(paper_text, readme, metadata) 
        system_prompt = self.get_system_prompt() 
        return prompt, system_prompt 
    
    def process_guidelines(self, schema_name = None):
        guidelines = {}
        if schema_name is not None:
            for g in open(f"guidelines/{schema_name}.md").read().split("\n")[2:]:                
                value = g.split("**")[-1].replace(":", "").strip()
                key = g.split("**")[1]
                guidelines[key] = value
        else:
            if self.version == "3.0":
                for key in self.schema.keys():
                    guidelines[key] = self.schema[key]['description']
            else:
                for key in self.schema.keys():
                    guidelines[key] = self.schema[key]['question']
        return guidelines

    def schema_to_template(self):
        # https://github.com/numindai/nuextract/tree/main
        type_mapper = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "url": "string",
            "year": "integer",
            "bool": [True, False],
            "list[str]": "multi-label"
        }
        template = {}
        for key in self.schema.keys():
            type = self.schema[key]['answer_type']
            if 'options' in self.schema[key]:
                options = self.schema[key]['options']
            else:
                options = None
            if type in ['str', 'url', 'year']:
                template[key] = options if options is not None else type_mapper[type]
            if type == 'int':
                template[key] = "integer"
            if type == 'float':
                template[key] = "number"
            if type == 'bool':
                template[key] = [True, False]
            if type == 'list[str]':
                template[key] = [options] if options is not None else type_mapper[type]
            if 'dict' in type:
                columns = type.split('dict[')[1].split(']')[0].split(',')
                columns = [column.strip() for column in columns]
                results = {}
                for column in columns:
                    if column in self.schema:
                        results[column] = type_mapper[self.schema[column]['answer_type']]
                template[key] = [results]
                    
        return json.dumps(template, indent=4)
    
    def mole_to_mextract(self):
        for key in self.schema.keys():
            if "question" in self.schema[key]:
                del self.schema[key]["question"]
            if "validation_group" in self.schema[key]:
                del self.schema[key]["validation_group"]
            if "option_description" in self.schema[key]:
                del self.schema[key]["option_description"]
            if self.schema[key]["answer_type"] == "date[year]":
                self.schema[key]["answer_type"] = "year"
            if self.schema[key]["answer_type"] == "List[str]":
                self.schema[key]["answer_type"] = "list"
            if "List[Dict" in self.schema[key]["answer_type"]:
                self.schema[key]["answer_type"] = self.schema[key]["answer_type"].replace("List[Dict", "list[dict")
        return self.schema
    
    def mole_to_slot(self):
        #TODO descriptions need some fixing ... venue title and veneue names are mixed ups 
        type_mapper = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "url": "string",
            "date[year]": "integer",
            "bool": "boolean"
        }
        descriptions = self.guidelines
        properties = {}
        for key in self.schema.keys():
            type = self.schema[key]['answer_type']
                
            if type == 'List[str]':
                properties[key] = {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            elif 'List[Dict' in type:
                columns = type.split('Dict[')[1].split(']')[0].split(',')
                columns = [column.strip() for column in columns]
                object_type = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {}
                    }
                }
                for column in columns:
                    if column in self.schema:
                        column_type = self.schema[column]['answer_type']
                        if column_type == 'List[str]':
                            object_type["items"]["properties"][column] = { # don't allow complex data types inside objects, this is a result of ill-defined Lanuage in the MultiSchema.
                                "type": "string",
                            }
                        else:
                            object_type["items"]["properties"][column] = {
                                "type": type_mapper[column_type]
                            }

                properties[key] = object_type
            else:
                properties[key] = {
                    "type": type_mapper[type]
                }
            
            if 'options' in self.schema[key]:
                if "items" in properties[key]:
                    properties[key]["items"]["enum"] = self.schema[key]['options']
                else:
                    properties[key]["enum"] = self.schema[key]['options']
            if key in descriptions:
                properties[key]["description"] = descriptions[key]          
        return properties
    
    
    def get_mole_schema(self):
        schema_name = self.get_schema_name()
        return json.load(open(f"schema/{schema_name}.json"))
    
    def dict(self):
        return json.loads(self.schema())
    
    def json(self):
        return json.loads(self.model_dump_json())

    def get_key_type(self, key):
        answer_type = None
        answer_min = 0
        answer_max = ANSWER_MAX
        options = None
        if self.version == "3.0":
            answer_type = self.schema[key]['type']
            for min_key in ['minLength', 'minimum', 'minItems']:
                if min_key in self.schema[key]:
                    answer_min = self.schema[key][min_key]
                    break
            for max_key in ['maxLength', 'maximum', 'maxItems']:
                if max_key in self.schema[key]:
                    answer_max = self.schema[key][max_key]
                    break
            if 'enum' in self.schema[key]:
                options = self.schema[key]['enum']
        else:
            answer_type = self.schema[key]['answer_type']
            if "answer_min" in self.schema[key]:
                answer_min = self.schema[key]['answer_min']
            if "answer_max" in self.schema[key]:
                answer_max = self.schema[key]['answer_max']
            if 'options' in self.schema[key]:
                options = self.schema[key]['options']
            
        if answer_type in ["str", "string"]:
            key_type =  "string"
        elif answer_type in ["url"]:
            key_type = "url"
        elif answer_type in ["date[year]", "year"]:
            key_type = "year"
        elif answer_type in ["float", "number"]:
            key_type = "float"
        elif answer_type in ["int", "integer"]:
            key_type = "int"
        elif answer_type in ["bool", "boolean"]:
            key_type = "bool"
        elif answer_type in ["list", "array"] or "list" in answer_type.lower():
            key_type = "list"
        elif answer_type == "object":
            key_type = "dict"
        else:
            raise ValueError(f"Invalid answer type: {answer_type}")
        # print(key)
        # print(key_type)
        # print(answer_min)
        # print(answer_max)
        # print(options)
        
        return get_type(key_type)(answer_min = answer_min, answer_max = answer_max, options = options)
    
    def get_default(self, key):
        t = self.get_key_type(key)
        return t.get_default()

    def get_formatted_guidelines(self):
        formatted_guidelines = ""
        for i, key in enumerate(self.guidelines):
            formatted_guidelines += f"{i+1}. **{key}**: {self.guidelines[key]}\n"
        return formatted_guidelines

    def get_system_prompt(self):
        if self.version == "3.0":
            system_prompt = f"""
                You are a professional metadata extractor of datasets from research papers. 
                You will be provided 'Paper Text', 'Input Schema' and you must respond with an 'Output JSON'.
                The 'Output JSON' is a JSON with key:answer where the answer retrieves an attribute of the 'Input Schema' from the 'Paper Text'. 
                Each attribute in the 'Input Schema' has the following fields:
                - "type": The return type of the attribute, which is a value from [string, number, integer, list, boolean, object, array, null]
                - "description": A description of the attribute
                - "enum" (optional): A list of possible values for the attribute.
                The 'Output JSON' is a JSON that can be parsed using Python `json.load()`. USE double quotes "" not single quotes '' for the keys and values.
                The 'Output JSON' must have ONLY the keys in the 'Input Schema'.
            """
        else:
            system_prompt = f"""
                You are a professional metadata extractor of datasets from research papers. 
                You will be provided 'Paper Text', 'Input Schema' and you must respond with an 'Output JSON'.
                The 'Output JSON' is a JSON with key:answer where the answer retrieves an attribute of the 'Input Schema' from the 'Paper Text'. 
                Each attribute in the 'Input Schema' has the following fields:
                'options' : If the attribute has 'options' then the answer must be at least one of the options.
                'answer_type': The output type represents the type of the answer.
                'answer_min' : The minimum length of the answer depending on the 'answer_type'.
                'answer_max' : The maximum length of the answer depending on the 'answer_type'.
                The 'Output JSON' is a JSON that can be parsed using Python `json.load()`. USE double quotes "" not single quotes '' for the keys and values.
                The 'Output JSON' must have ONLY the keys in the 'Input Schema'.
            """
        if self.version == "2.0":
            system_prompt += "Use the following guidelines to extract the answer from the 'Paper Text':\n\n"
            system_prompt += self.get_formatted_guidelines()
        return system_prompt
        
    def evaluate_length(self, metadata):
        accuracy = 0
        attributes = self.get_attributes()
        # print(schema)
        for key in attributes:
            t  = self.get_key_type(key)
            length = t.validate_length(metadata[key])
            # if length < 1:
            #     print(key)
            #     print(t.answer_min)
            #     print(t.answer_max)
            accuracy += length
        return accuracy / len(attributes)
    
    def modify_length(self, length_constrain = 'low', accepted=True):
        metadata = self.model_dump()
        schema = json.loads(self.__class__.schema(length_constrain=length_constrain))
        for key in self.get_attributes():
            type  = self.get_answer_object(key)
            answer_min = schema[key]['answer_min']
            answer_max = -1 if "answer_max" not in schema[key] else schema[key]['answer_max']
            modified_value = type.modify_length(metadata[key], answer_min, answer_max, accepted)
            metadata[key] = modified_value
        return metadata
    
    def evaluate(self, metadata, gold_metadata, return_metrics_only = False, return_precision_only = False, exact_match = False):
        metadata = self.validate(metadata)
        results = {}
        # print(self.get_attributes())
        for key in self.get_attributes():
            results[key] = self.match_attributes(key, gold_metadata[key], metadata[key], exact_match = exact_match)

        precision = sum(results.values()) / len(results)
        if return_precision_only:
            return {'precision': precision}
        annotations_from_paper = gold_metadata['annotations_from_paper']
        annotated_attributes = [key for key in self.get_attributes() if key in annotations_from_paper and annotations_from_paper[key]]
        recall = sum([value for key, value in results.items() if key in annotated_attributes]) / len(annotated_attributes)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        length = self.evaluate_length(metadata)
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        results['length'] = length
        if return_metrics_only:
            return {'precision': precision, 'recall': recall, 'f1': f1, 'length': length}
        return results

    def match_attributes(self, key, attr1, attr2, exact_match = False):
        t = self.get_key_type(key)
        return t.compare(attr1, attr2, exact_match = exact_match)
    
    def get_random(self, key):
        t = self.get_key_type(key)
        return t.get_random()
    
    def generate_metadata(self, method = 'random'):
        metadata = {}
        for key in self.get_attributes():
            if method == 'random':
                metadata[key] = self.get_random(key)
            elif method == 'default':
                metadata[key] = self.get_default(key)
            else:
                raise ValueError(f"Invalid method: {method}")
        return metadata

    
    def validate(self, metadata):
        schema_attributes = self.get_attributes()
        all_attributes = schema_attributes + list(metadata.keys())
        for key in all_attributes:
            if key not in schema_attributes: # if the key is not in the schema, then delete it
                del metadata[key]
                continue
            t = self.get_key_type(key)
            if key not in metadata: # if the key is not in the data, then set it to the default
                metadata[key] = t.get_default()
            else:
                kd = metadata[key]
                if kd is None:
                    metadata[key] = t.get_default()
                else:
                    try:
                        metadata[key] = t.cast(kd)
                    except:
                        metadata[key] = t.get_default()
        return metadata
