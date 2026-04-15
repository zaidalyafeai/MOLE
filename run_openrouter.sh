for schema_name in ar en jp ru fr multi model tool s2orc bib; do
    echo "🚀 Starting evaluation for $schema_name..."
    uv run src/evaluate.py \
        --split test \
        --backend openrouter \
        --model  google/gemma-4-31b-it\
        --schema_name $schema_name \
        --version 3.0 \
        --max_model_len 8192 \
        --results_path results_comparison_finetuned \
        --format pdf_docling \
        --max_output_len 2048 \
        --overwrite
done