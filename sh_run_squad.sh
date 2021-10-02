#Intensive module
export SQUAD_DIR=data/visquad-v1
export TRAIN_FILE=train.json
export DEV_FILE=dev.json
python ./run_squad.py \
    --model_type xlm-roberta \
    --model_name_or_path visquad-v1/xlm-roberta-base \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/$TRAIN_FILE \
    --predict_file $SQUAD_DIR/$DEV_FILE \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --max_query_length=128 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=16 \
    --warmup_steps=814 \
    --output_dir visquad-v1/xlm-roberta-base \
    --save_steps 2500 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --overwrite_output_dir \
    --fp16 \
    --overwrite_output_dir 
