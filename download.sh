FULL_MODEL_NAME=$1
# split model name
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d '/' -f 2)
uv run hf download $FULL_MODEL_NAME --local-dir $MODEL_NAME