# type: ignore

from schema import Schema
from type_classes import *
from rich import print
import json

schema = Schema(schema_name = "ar")
print(schema.dump_schema())
print(schema.get_formatted_guidelines())
raise
gold_metadata1  = {
    "Name": "ahmad",
    "Age": 20,
    "Website": "https://www.google.com",
    "Hobbies": ["reading"],
    'Married': True,
    "Sons":[],
    "annotations_from_paper": {
        "Name": 1,
        "Age": 1,
        "Website": 1,
        "Hobbies": 1,
        "Sons": 1,
        "Married": 1
    }
}

parent_schema = {
        "Name": {"type": "string", "description": "Name of the person", "minLength": 1, "maxLength": 1},
        "Age": {"type": "integer", "description": "Age of the person", "minimum": 0, "maximum": 100},
        "Website": {"type": "string", "description": "Website of the person", "minLength": 1, "maxLength": 1},
        "Hobbies": {"type": "array", "description": "Hobbies of the person", "items": {"type": "string", "enum": ["reading", "swimming", "coding"]}, "minItems": 1, "maxItems": 3},
        "Married": {"type": "boolean", "description": "Married status of the person"},
        "Sons": {"type": "array", "description": "Sons of the person", "items": {"type": "object"}, "minItems": 0, "maxItems": 3},
}
sc = Schema(
    schema = parent_schema,
)
metadata = json.load(open('testfiles/test1.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test1')


# [reading] - > [reading, swimming]
metadata = json.load(open('testfiles/test2.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)
assert evaluation_results['Hobbies'] == 0.5, f'❌ Hobbies value should be 0.5 but got {evaluation_results["Hobbies"]}'

print('✅ passed test2')

metadata = json.load(open('testfiles/test3.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)
assert evaluation_results['Age'] == 0, f'❌ Age should be 0 but got {evaluation_results["Age"]}'
print('✅ passed test3')

metadata = json.load(open('testfiles/test4.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)
assert abs(evaluation_results['length'] - 0.83) < 0.01, f'❌ length should be 0.83 but got {evaluation_results["length"]}'
print('✅ passed test4')


metadata = json.load(open('testfiles/test5.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test5')


gold_metadata2 = {
    "Name": "ahmad",
    "Age": 20,
    "Website": "https://www.google.com",
    "Hobbies": ["swimming", "coding", "reading"],
    'Married': True,
    "Sons": [
            {"Name": "ahmad", "Age": 20},
            {"Name": "ali", "Age": 10}
    ],
    "annotations_from_paper": {
        "Name": 1,
        "Age": 1,
        "Website": 1,
        "Hobbies": 1,
        "Sons": 1,
        "Married": 1
    }
}


metadata = json.load(open('testfiles/test6.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata2)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test6')


metadata = json.load(open('testfiles/test7.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test7')

metadata = json.load(open('testfiles/test8.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test8')


metadata = json.load(open('testfiles/test9.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test9')

metadata = json.load(open('testfiles/test10.json'))
evaluation_results = sc.evaluate(metadata, gold_metadata1)

for m in evaluation_results:
    assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test10')

default_metadata = {
    "Name": "",
    "Age": 0,
    "Website": "",
    "Hobbies": [],
    "Sons":[],
    'Married': False,
    "annotations_from_paper": {
        "Name": 1,
        "Age": 1,
        "Website": 1,
        "Hobbies": 1,
        "Sons": 1,
        "Married": 1
    }
}
metadata = sc.generate_metadata(method = 'default')
evaluation_results = sc.evaluate(metadata, default_metadata, return_metrics_only=True)
for m in evaluation_results:
    if m == 'length':
        assert abs(evaluation_results[m] - 0.5) < 0.01, f'❌ {m} value should be 0.5 but got {evaluation_results[m]}'
    else:
        assert evaluation_results[m] == 1, f'❌ {m} value should be 1 but got {evaluation_results[m]}'
print('✅ passed test11')

# validate metadata
from jsonschema import validate
metadata = json.load(open('testfiles/test12.json'))
validate(instance=metadata, schema=sc.schema)
print('✅ passed test12')

# validate metadata
from jsonschema import validate
sc = Schema(schema_name="ar")
metadata = json.load(open('testfiles/test11.json'))

validate(instance=metadata, schema=sc.schema)
print('✅ passed test13')

# validate metadata
from jsonschema import validate
schema = Schema(schema_name = 'ar', version = "3.0")
metadata = json.load(open('testfiles/test13.json'))
validate(instance=metadata, schema=schema.schema)
print('✅ passed test14')

sc = Schema(schema_name = 'ar', version = "2.0")
gold_metadata = json.load(open('testfiles/test11.json'))
metadata = json.load(open('testfiles/test11.json'))
del metadata["annotations_from_paper"]
results = sc.evaluate(metadata, gold_metadata)
print('✅ passed test15')

parent_schema["Car"] = {
    "type": "object",
    "description": "Car of the person",
    "properties": {
        "Brand": {"type": "string"},
        "Model": {"type": "string"},
        "Year": {"type": "integer"}
    }
}
sc = Schema(schema = parent_schema)
gold_metadata = json.load(open('testfiles/test16.json'))
metadata = json.load(open('testfiles/test16.json'))
del metadata["annotations_from_paper"]
results = sc.evaluate(metadata, gold_metadata)
print('✅ passed test16')
