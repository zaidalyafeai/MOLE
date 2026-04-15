r=8
alpha=16
model_name=$1
max_train_examples=$3
wandb=$2_$3
data_path=$4
num_gpus=$5
max_model_len=$6
max_output_len=$7
output_dir="lora_tuning_adapters_${model_name}-${wandb}/r_${r}_alpha_${alpha}"
                
if [ -d "$output_dir" ]; then
    echo "Skipping model=$model_name, r=$8, alpha=$alpha (directory already exists: $output_dir)"
else
    echo "Submitting model=$model_name, r=$r, alpha=$alpha"
    sbatch --mem=16GB \
            --gres=gpu:v100:$num_gpus \
            --time=6:00:00 \
            --output=${output_dir}/slurm_run.out \
            --error=${output_dir}/slurm_run.err \
            --wrap="CUDA_LAUNCH_BLOCKING=1 uv run python src/finetune_hf.py --output_dir $output_dir --lora_r $r --lora_alpha $alpha --model_name $model_name --wandb_name $wandb --max_train_examples $max_train_examples --data_path $data_path --format pdf_docling --max_model_len $max_model_len --max_output_len $max_output_len"
fi

        
 
