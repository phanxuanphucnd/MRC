#sketchy module
export DATA_DIR=data/uit-visquad
export TASK_NAME=squad
python run_ev.py \
    --model_type phobert \
    --model_name_or_path vinai/phobert-base \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 256 \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=8 \
    --warmup_steps=814 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --eval_all_checkpoints \
    --output_dir models/cls-phobert \
    --save_steps 2500 \
    --overwrite_output_dir