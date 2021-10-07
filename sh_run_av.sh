#Intensive module
export SQUAD_DIR=data/uit-visquad
export TRAIN_FILE=train-0.json
export DEV_FILE=dev-0.json
export PREDICT_FILE=ptest.json
python ./run_av.py \
    --model_type xlm-roberta \
    --model_name_or_path xlm-roberta-large \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/$TRAIN_FILE \
    --predict_file $SQUAD_DIR/$DEV_FILE \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=16 \
    --warmup_steps=814 \
    --output_dir models/uit-visquad/xlm-roberta-large-av  \
    --eval_all_checkpoints \
    --save_steps 2500 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --overwrite_output_dir 
