export CUDA_VISIBLE_DEVICES=0,1,2,3
for((i=0;i<10;i++));  
do   

python run_xlnet.py \
--model_type xlnet \
--model_name_or_path ./pretrained_model/xlnet_mid \
--do_train \
--do_eval \
--do_test \
--data_dir ./data_re_10fold_firstpl/data_$i \
--output_dir ./results/xlnet_mid_01/xlnet_mid$i \
--max_seq_length 256 \
--split_num 3 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 64 \
--learning_rate 2e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 5000 \
--report_steps 200 ;

done  





