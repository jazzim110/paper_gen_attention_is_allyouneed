import sentencepiece as spm
from collections import Counter, OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from jaehoon.sentence_model_gen import Sentence_model_gen



class Preprocess:

    def __init__(self, config, logger):
        self.prefix_en = "EN"
        self.prefix_de = "DE"
        self.config = config
        self.logger = logger
        self.sentence_model_gen = config.sentence_model_gen
        self.input_file = config.input_file_dir
        self.output_file = config.output_file_dir
        self.batch_size = config.batch_size

        if self.sentence_model_gen == "True":
            Sentence_model_gen(self.prefix_en, self.prefix_de, config)


    def tokenize_en(self):

        token2idx_en = dict()
        idx2token_en = dict()
        wordpiece_list_en = list()
        word_maxlen_en = 0
        sen_maxlen_en = 0
        sen_len_list_en = list()

        sp_en = spm.SentencePieceProcessor()
        sp_en.Load('{}.model'.format(self.prefix_en))

        with open(self.input_file, 'r', encoding='utf-8') as df:
            lines = df.readlines()
            for i, line in enumerate(lines):
                wordpiece = sp_en.EncodeAsPieces(line)
                wordidx = sp_en.EncodeAsIds(line)
                wordpiece_list_en.append(wordidx)
                sen_len_list_en.append(len(line))
                if len(wordpiece) >= sen_maxlen_en:
                    sen_maxlen_en = len(wordpiece)
                for word, idx in zip(wordpiece, wordidx):
                    token2idx_en[word] = idx
                    idx2token_en[idx] = word
                    if len(word) >= word_maxlen_en:
                        word_maxlen_en = len(word)
                if i == 0:
                    print("first wordpiece : {}".format(wordpiece))
                    print("first wordidx : {}".format(wordidx))
                    print("first sentense length : {}".format(len(line)))
                    print("dictory : {}".format(token2idx_en))

                    self.logger.info("----- tokenize en first data -----")
                    self.logger.info(wordpiece)
                    self.logger.info(wordidx)
                    self.logger.info(token2idx_en)

        return token2idx_en, idx2token_en, wordpiece_list_en, word_maxlen_en, sen_maxlen_en, sen_len_list_en


    def tokenize_de(self):

        token2idx_de = dict()
        idx2token_de = dict()
        wordpiece_list_de = list()
        word_maxlen_de = 0
        sen_maxlen_de = 0
        sen_len_list_de = list()

        sp_de = spm.SentencePieceProcessor()
        sp_de.Load('{}.model'.format(self.prefix_de))

        with open(self.output_file, 'r', encoding='utf-8') as df:
            lines = df.readlines()
            for i, line in enumerate(lines):
                wordpiece = sp_de.EncodeAsPieces(line)
                wordidx = sp_de.EncodeAsIds(line)
                wordpiece_list_de.append(wordidx)
                sen_len_list_de.append(len(line))
                if len(wordpiece) >= sen_maxlen_de:
                    sen_maxlen_de = len(wordpiece)
                for word, idx in zip(wordpiece, wordidx):
                    token2idx_de[word] = idx
                    idx2token_de[idx] = word
                    if len(word) >= word_maxlen_de:
                        word_maxlen_de = len(word)
                if i == 0:
                    print("first wordpiece : {}".format(wordpiece))
                    print("first wordidx : {}".format(wordidx))
                    print("first sentense length : {}".format(len(line)))
                    print("dictory : {}".format(token2idx_de))

                    self.logger.info("----- tokenize de first data -----")
                    self.logger.info(wordpiece)
                    self.logger.info(wordidx)
                    self.logger.info(token2idx_de)

        return token2idx_de, idx2token_de, wordpiece_list_de, word_maxlen_de, sen_maxlen_de, sen_len_list_de


    def get_final_data(self):

        source = list()
        target = list()

        token2idx_en, idx2token_en, wordpiece_list_en, word_maxlen_en, sen_maxlen_en, sen_len_list_en = self.tokenize_en()
        token2idx_de, idx2token_de, wordpiece_list_de, word_maxlen_de, sen_maxlen_de, sen_len_list_de = self.tokenize_de()

        self.logger.info("max length of en word : {}".format(word_maxlen_en))
        self.logger.info("max length of de word : {}".format(word_maxlen_de))
        self.logger.info("max length of en sentence : {}".format(sen_maxlen_en))
        self.logger.info("max length of de sentence : {}".format(sen_maxlen_de))

        cutoff_max_sen_len_en = self.find_cutoff_max_sen_len(0.1, sen_len_list_en)
        cutoff_max_sen_len_de = self.find_cutoff_max_sen_len(0.1, sen_len_list_de)
        cutoff_max_sen_len = max(cutoff_max_sen_len_en, cutoff_max_sen_len_de)

        print(cutoff_max_sen_len)

        for idx in range(len(wordpiece_list_en)):
            if max(len(wordpiece_list_en[idx]), len(wordpiece_list_de[idx])) <= cutoff_max_sen_len:
                source.append(np.array(wordpiece_list_de[idx]))
                target.append(np.array(wordpiece_list_en[idx]))

        print(len(source))
        print(len(target))

        inputs = np.zeros([len(source), cutoff_max_sen_len], dtype=np.int32)
        outputs = np.zeros([len(target), cutoff_max_sen_len], dtype=np.int32)

        for idx, (x, y) in enumerate(zip(source, target)):  # source : english, target : german
            inputs[idx, :len(x)] = x
            outputs[idx, :len(y)] = y

        print("Source Matrix Shape (DE):", inputs.shape)
        print("Target Matrix Shape (EN):", outputs.shape)
        print()
        print('------------------------ Show the example case ------------------------')
        print(inputs[0])
        print(outputs[0])

        print('------------------------ Show the example case ------------------------')
        print(inputs[10])
        print(outputs[10])

        self.logger.info("inputs / outputs shape, first : inputs / second : outputs")
        self.logger.info(inputs.shape)
        self.logger.info(outputs.shape)

        return token2idx_en, idx2token_en, token2idx_de, idx2token_de, inputs, outputs


    def find_cutoff_max_sen_len(self, cutoff_value, sen_len_list):

        count_sen_len = Counter(sen_len_list)
        total_count_sen_len = sum(count_sen_len.values())
        dict_count_sen_len = dict(count_sen_len)
        dict_count_sen_len = OrderedDict(sorted(dict_count_sen_len.items()))

        temp_count_sen_len = 0
        for k, v in dict_count_sen_len.items():
            temp_count_sen_len += v
            cdf_inv = temp_count_sen_len / total_count_sen_len
            if cdf_inv >= cutoff_value:
                cutoff_max_sen_len = k
                print("누적 {}%를 차지하는 sentence length 값 : {}".format(cutoff_value*100, cutoff_max_sen_len))
                break

        self.logger.info("max sentence length cutoff value")
        self.logger.info(cutoff_max_sen_len)

        return cutoff_max_sen_len


    def dataloader(self, inputs, outputs, batch_size):

        X = DataLoader(inputs, batch_size=batch_size, drop_last=True)
        Y = DataLoader(outputs, batch_size=batch_size, drop_last=True)

        return X, Y


    def preprocessing(self):

        token2idx_en, idx2token_en, token2idx_de, idx2token_de, inputs, outputs = self.get_final_data()
        X, Y = self.dataloader(inputs, outputs, self.batch_size)

        self.logger.info("dictionary shape")
        self.logger.info("length of token2idx_en : {}".format(len(token2idx_en)))
        self.logger.info("length of token2idx_de : {}".format(len(token2idx_de)))

        return X, Y, token2idx_en, idx2token_en, token2idx_de, idx2token_de








