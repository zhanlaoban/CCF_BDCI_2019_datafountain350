export CUDA_VISIBLE_DEVICES=0,1,2,3
for((i=0;i<10;i++));  
do   

python run_bert.py \
--model_type bert \
--model_name_or_path ./pretrained_model/bert_base \
--do_train  \
--do_eval \
--do_test \
--data_dir ./data_re_10fold_secondpl/data_$i \
--output_dir ./results/bert_base_04/bert_base$i \
--max_seq_length 256 \
--split_num 3 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 2e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 10000 

done  





