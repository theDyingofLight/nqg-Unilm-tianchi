import json, os, re

root = r"."
TRAIN_ORIGIN_FILE = root + '/user_data/tmp_data/round1_train_0907_origin.json'
TRAIN_FILE = root + '/user_data/tmp_data/round1_train_0907.json'
TEST_FILE = root + '/tcdata/juesai.json'
DEV_FILE = root + '/user_data/tmp_data/round1_dev_0907.json'
RESULT_FILE = root + '/result.json'

train_origin_data = root + '/user_data/tmp_data/train_origin_data.json'
train_data = root + '/user_data/tmp_data/train_data.json'
dev_data = root + '/user_data/tmp_data/dev_data.json'
test_data = root + '/user_data/tmp_data/juesai_data.json'
pred_data = root + '/user_data/tmp_data/predict_.json'

train_append_data = root + '/user_data/tmp_data/train_append_data.json'
train_new_data = root + '/user_data/tmp_data/train_new_data.json'

max_passage_len = 440
max_question_len = 69


def transfer_test(pred, test_origin, test_file):
    with open(pred, 'r', encoding='utf-8') as pred:
        test_tgt = pred.readlines()
    with open(test_origin, 'r', encoding='utf-8') as test_origin_file:
        test_origin_data = test_origin_file.readlines()
        test_origin_data = [json.loads(line) for line in test_origin_data]
    with open(test_file, 'r', encoding='utf-8') as test:
        test_data = json.load(test)
        count = 0
        for each in test_data:
            for qa in each["annotations"]:
                ans = " ".join(qa["A"].replace('\"', '').replace('\\n', '').split("\n"))
                test_ans = test_origin_data[count]["answer_text"]
                if test_origin_data[count]["answer_text"] == ans:
                    qa['Q'] = test_tgt[count].strip()
                    count += 1
                else:
                    print("not match")
                    qa['Q'] = test_tgt[count].strip()
                    count += 1
        print(count)
        json.dump(test_data, open(RESULT_FILE, 'w', encoding='utf-8'), ensure_ascii=False)


transfer_test(pred_data, test_data, TEST_FILE)
print("process data to result ok")
