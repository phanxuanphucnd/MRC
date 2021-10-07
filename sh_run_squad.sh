#Intensive module
export SQUAD_DIR=data/uit-visquad
export TRAIN_FILE=train-0.json
export DEV_FILE=dev-0.json
export PREDICT_FILE=ptest.json
python ./run_squad.py \
    --model_type xlm-roberta \
    --model_name_or_path xlm-roberta-large \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/$TRAIN_FILE \
    --predict_file $SQUAD_DIR/$PREDICT_FILE \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --max_query_length=128 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=16 \
    --warmup_steps=814 \
    --eval_all_checkpoints \
    --output_dir models/uit-visquad/xlm-roberta-large \
    --save_steps 2500 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --fp16 \
    --overwrite_output_dir 
