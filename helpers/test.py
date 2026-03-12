import json
from jsonschema import validate, ValidationError

# Schema allowing only defined properties
schema_strict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"}
    },
    "required": ["name", "price"],
    "additionalProperties": False # Explicitly forbid extra fields
}

# Schema allowing extra properties
schema_flexible = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"}
    },
    "required": ["name", "price"],
    # additionalProperties defaults to True, so we can omit it
}

valid_data = {"name": "Eggs", "price": 34.99}
invalid_data_extra_field = {"name": "Eggs", "price": 34.99, "extra_field": "some value"}

# Validation with strict schema
try:
    validate(instance=valid_data, schema=schema_strict)
    print("Strict schema: Valid data passes.")
except ValidationError as e:
    print(f"Strict schema: Valid data fails! {e.message}")

try:
    validate(instance=invalid_data_extra_field, schema=schema_strict)
    print("Strict schema: Invalid data passes (unexpected).")
except ValidationError as e:
    print(f"Strict schema: Invalid data fails as expected: {e.message}")

# Validation with flexible schema
try:
    validate(instance=invalid_data_extra_field, schema=schema_flexible)
    print("Flexible schema: Invalid data with extra field passes as expected.")
except ValidationError as e:
    print(f"Flexible schema: Invalid data with extra field fails! {e.message}")
