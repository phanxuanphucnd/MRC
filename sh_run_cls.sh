#sketchy module
export DATA_DIR=data/uit-visquad
export TRAIN_FILE=train-0.json
export DEV_FILE=dev-0.json
export PREDICT_FILE=ptest.json
export TASK_NAME=squad
python run_ev.py \
    --model_type xlm-roberta \
    --model_name_or_path xlm-roberta-large \
    --task_name $TASK_NAME \
    --train_file $TRAIN_FILE \
    --dev_file $DEV_FILE \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 384 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=16 \
    --warmup_steps=814 \
    --learning_rate 2e-5 \
    --num_train_epochs 15 \
    --eval_all_checkpoints \
    --output_dir models/cls/xlm-roberta-cls \
    --save_steps 2500 \
    --overwrite_output_dir \
    --fp16