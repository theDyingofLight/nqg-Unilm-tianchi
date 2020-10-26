# NQG-Unilm for 天池中医文献问题生成任务

源代码来自于：https://github.com/YunwenTechnology/Unilm

数据来源：https://tianchi.aliyun.com/competition/entrance/531826/information

预训练模型ROBERTA（12层RoBERTa模型）：https://github.com/brightmart/roberta_zh



## Dependencies

This code is written in Python. Dependencies include

- python >= 3.6

- pytorch >= 1.4

- nltk

- tqdm

- transformers 2.6.0

  

## 改进

1. 对数据进行预处理：修正由于答案片段中标点符号中英文未切换产生的答案不在原文中的数据问题，然后按照预先规定的长度，若长度超过规定则截取答案附近一段原文；另外由于该数据集特点，答案和提问的内容联系比较紧密，可以只看有限区域中的原文信息，于是对原文进行分句，较短答案则只需要上下文级答案所在句子的三句话，对原文信息进行精简。最后得到一个可以用于输入的预处理数据。

2. 模型：UNILM

   使用roberta-l12预训练模型，手动更改一些[unused]为原词典中未出现的词

   在embedding层增加了answer-tag，定位分词后答案在原文中的位置，并得到一个和原文大小相同但只有答案在的地方才为1，没有答案的地方则标记为0的answer-tag向量，用于增加原文embedding的信息。

   修改生成阶段代码，以减少生成所需时间。

   

## Configuration

```shell
python process_data.py

python -u run_seq2seq.py --data_dir ./userdata/tmp_data/ --src_file train_data.json --model_type unilm --model_name_or_path RoBERTa_zh_L12_PyTorch/ --output_dir output_dir/ --max_seq_length 512 --max_position_embeddings 512 --do_train --do_eval --do_lower_case --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20 --mask_prob 0.7 --label_smoothing 0.1 --hidden_dropout_prob 0.4

python -u decode_seq2seq.py --model_type unilm --model_name_or_path RoBERTa_zh_L12_PyTorch/ --model_recover_path ./output_dir/model.25.bin --max_seq_length 512 --input_file ./user_data/tmp_data/juesai_data.json --output_file ./user_data/tmp_data/predict_.json --do_lower_case --batch_size 16 --beam_size 8 --max_tgt_length 69

python process_to_result.py
```

模型下载：[百度网盘](https://pan.baidu.com/s/1maQPZF_SBuZ-t4xEJ9zjHQ) 提取码：z433



## Results

rouge-l

初赛成绩：0.6309

复赛成绩：0.6242