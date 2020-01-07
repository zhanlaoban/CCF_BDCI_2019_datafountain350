#convert tensorflow model in tf_chinese_roberta_wwm_ext to pytorch model
python pytorch_transformers.convert_xlnet_checkpoint_to_pytorch \
	--tf_checkpoint_path ./pretrained_model/xlnet_large/XLNet_zh_Large_L-24_H-1024_A-16 \
	--xlnet_config_file ./pretrained_model/xlnet_large/XLNet_zh_Large_L-24_H-1024_A-16/xlnet_config.json \
	--pytorch_dump_folder_path ./pretrained_model/xlnet_large/pytorch_model.bin \
#	--bert_config_file tf_chinese_roberta_wwm_ext/bert_config.json
