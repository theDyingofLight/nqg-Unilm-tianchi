#!usr/bin/python
import json, os, re

# root = r'/raid/wsy/competition/Unilm-master'
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_utils import PreTrainedModel
from pytorch_transformers.configuration_utils import PretrainedConfig
import torch
from torch.nn.parameter import Parameter
from modeling_unilm import UnilmForSeq2Seq, UnilmConfig

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

resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


def splitsentence(sentence):
    s = sentence
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def process_file(data_file):
    with open(data_file, 'r', encoding="utf-8") as file:
        content = json.load(file)
        articles_all = []
        questions_all = []
        answers_all = []
        for each in content:
            article = ' '.join(each["text"].replace('\"', '').replace('\\n', '').split("\n"))
            annotation = each["annotations"]
            for qa in annotation:
                articles_all.append(article)
                questions_all.append(" ".join(qa["Q"].replace('\"', '').replace('\\n', '').split("\n")))
                answers_all.append(" ".join(qa["A"].replace('\"', '').replace('\\n', '').split("\n")))
    return articles_all, questions_all, answers_all


def normalize_passage(sentence, answer):
    index = sentence.find(answer)

    if index != -1:
        if len(sentence) >= max_passage_len:
            if len(answer) >= max_passage_len:
                answer = answer[:max_passage_len]
                sentence = answer
            else:
                cut_num = int((max_passage_len - len(answer)) / 2)
                index_prev = max(index - cut_num, 0)
                index_after = min(index + len(answer) + cut_num, len(sentence))
                sentence = sentence[index_prev:index_after]
    else:
        print("not find")
        if len(sentence) >= max_passage_len:
            sentence = sentence[:max_passage_len]
    return sentence, answer


def get_shorter_passage(sentence, answer, question):
    index = sentence.find(answer)
    ques = question
    if index != -1:
        sent_split = splitsentence(sentence)
        ans_split = splitsentence(answer)
        if len(ans_split) == 1:
            for i in range(len(sent_split)):
                if ans_split[0] in sent_split[i]:
                    prio = "" if i == 0 else sent_split[i - 1]
                    after = "" if i == len(sent_split) - 1 else sent_split[i + 1]
                    if len(prio) + len(after) + len(sent_split[i]) <= max_passage_len:
                        sentence = prio + " " + sent_split[i] + " " + after
                    else:
                        sentence = after + " " + sent_split[i] if len(after) + len(sent_split[i]) else sent_split[i]
                        sentence = prio + " " + sent_split[i] if len(prio) + len(sent_split[i]) else sent_split[i]
    return sentence, answer


def make_conll_format(sentences, questions, answers, file):
    instances = []
    for i in range(len(sentences)):
        sentence, answer = normalize_passage(sentences[i], answers[i])
        sentence, answer = get_shorter_passage(sentence, answer, questions[i])
        instance = {'src_text': sentence, 'tgt_text': questions[i], 'answer_text': answer}
        instances.append(instance)
    with open(file, "w", encoding='utf-8') as result:
        for each in instances:
            result.write(json.dumps(each, ensure_ascii=False) + '\n')


def append_train(append_data, tra_data):
    with open(append_data, 'r', encoding='utf-8') as app_file:
        app_data = app_file.readlines()
    with open(tra_data, 'r', encoding='utf-8') as tr_file:
        tr_data = tr_file.readlines()
    all_data = tr_data + app_data
    with open(train_new_data, 'w', encoding='utf-8') as result:
        for each in all_data:
            result.write(each)


def add_vocab(vocab, train, new_vocab):
    tokenizer = BertTokenizer.from_pretrained(
        r'F:\文本生成任务\competition-tianchi\Unilm-master\Unilm-master\MTBERT\vocab.txt')
    with open(vocab, 'r', encoding='utf-8') as vocab_file:
        word_vocab = vocab_file.readlines()
        word_vocab = [i.strip() for i in word_vocab]
    with open(train, 'r', encoding='utf-8') as train_file:
        content = json.load(train_file)
        for each in content:
            article = ' '.join(each["text"].replace('\"', '').replace('\\n', '').split("\n"))
            tokens = tokenizer.tokenize(article)
            for word in tokens:
                if word not in word_vocab:
                    word_vocab.append(word)
    with open(new_vocab, 'w', encoding='utf-8') as new_vocab_file:
        for i in range(len(word_vocab)):
            new_vocab_file.write(word_vocab[i] + '\n')
    print(len(word_vocab))


def change_pretrain(bert_origin):
    config = UnilmConfig.from_pretrained(bert_origin)
    model = UnilmForSeq2Seq.from_pretrained(
        bert_origin, state_dict=None, config=config)
    a = torch.randn((21296 - 21128, 768)) * (0.5 / 768)
    b = torch.randn((21296 - 21128)) * (0.5 / 768)
    c = torch.randn((21296 - 21128, 768)) * (0.5 / 768)
    d = torch.randn((21296 - 21128)) * (0.5 / 768)
    model.bert.embeddings.word_embeddings.weight = Parameter(torch.cat(
        (model.bert.embeddings.word_embeddings.weight.data, a), dim=0), requires_grad=True)
    model.bert.embeddings.word_embeddings.num_embeddings = 21296
    model.cls.predictions.bias = Parameter(torch.cat((model.cls.predictions.bias.data, b), dim=0), requires_grad=True)
    model.cls.predictions.decoder.weight = Parameter(torch.cat(
        (model.cls.predictions.decoder.weight.data, c), dim=0), requires_grad=True)
    model.cls.predictions.decoder.bias = Parameter(torch.cat((model.cls.predictions.decoder.bias.data, d), dim=0),
                                                   requires_grad=True)
    model.cls.predictions.decoder.out_features = 21296

    model.save_pretrained(root + r'\new_mtbert')

'''
sentence_all, question_all, answers_all = process_file(TRAIN_FILE)
make_conll_format(sentence_all, question_all, answers_all, train_data)
dev_sentence_all, dev_question_all, dev_answers_all = process_file(DEV_FILE)
make_conll_format(dev_sentence_all, dev_question_all, dev_answers_all, dev_data)
test_sentence_all, test_question_all, test_answers_all = process_file(TEST_FILE)
make_conll_format(test_sentence_all, test_question_all, test_answers_all, test_data)
print("process data ok")
'''
