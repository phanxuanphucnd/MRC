#Intensive module
export SQUAD_DIR=data/visquad-v1
export TRAIN_FILE=train.json
export DEV_FILE=dev.json
python ./run_av_seq_sc.py \
    --model_type phobert \
    --model_name_or_path vinai/phobert-base \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/$TRAIN_FILE \
    --predict_file $SQUAD_DIR/$DEV_FILE \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --max_seq_length 256 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=32 \
    --warmup_steps=814 \
    --output_dir visquad-v1/phobert-sc \
    --eval_all_checkpoints \
    --save_steps 10000 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --overwrite_output_dir 
