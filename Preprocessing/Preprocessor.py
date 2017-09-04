import random
import nltk


def CombineTokens(tokens):
    strg = reduce(lambda x, y: str(x) + ' ' + str(y), tokens)
    return strg


def regular_sample(index_record, percent):
    train_index = []
    test_index = []
    slice_len = percent[0] + percent[1]
    # slice_num = len(index_record) / slice_len + 1 if len(index_record) % slice_len > 0 else len(
    #     index_record) / slice_len
    # for index in range(slice_num):
    #     if index != slice_num - 1:
    #         train_index += index_record[slice_len * index:slice_len * index + percent[0]]
    #         test_index += index_record[slice_len * index + percent[0]:slice_len * index + percent[0] + percent[1]]
    #     else:
    #         pass
    for index in range(slice_len):
        if index < percent[0]:
            train_index += index_record[index::slice_len]
        else:
            test_index += index_record[index::slice_len]
    return train_index, test_index


class Preprocessor(object):
    def preprocess(self):
        print 'This is a preprocessor.'
        pass

    pass


class CRFPreprocessor(Preprocessor):
    def __init__(self, **kw):
        self.source_data_file = kw.get('source_data_file')
        self.train_file = kw.get('train_file')
        self.test_file = kw.get('test_file')

    def preprocess(self):
        print 'This is CRF preprocessor.'

    '''
        We have to load all of train data and test data in memory.
    '''

    def get_train_data(self, percent=(2, 1), isRandom=True):  # train size : test size == 2 :  1
        fopen = open(self.source_data_file)
        index_record = [index for index, line in enumerate(fopen) if not line.strip()]
        fopen.close()
        train_index = []
        test_index = []
        if isRandom:
            train_index = random.sample(index_record, len(index_record) / (percent[0] + percent[1]) * percent[0])
            test_index = list(set(index_record) - set(train_index))
        else:
            train_index, test_index = regular_sample(index_record, percent)
        # print index_record
        with open(self.source_data_file) as fopen_source, open(self.train_file, 'w') as fopen_train, open(
                self.test_file, 'w') as fopen_test:
            sents = []
            for index, line in enumerate(fopen_source):
                sents.append(line)
                if index in train_index:
                    fopen_train.write(''.join(sents))
                    sents = []
                elif index in test_index:
                    fopen_test.write(''.join(sents))
                    sents = []


'''
    Retag the null label with pos tag.
'''


class PreLabelWithPosPreprocessor(Preprocessor):
    def __init__(self, **kw):
        self.train_file = kw.get('train_file')
        self.test_file = kw.get('test_file')

    def preprocess(self):
        print 'This is PreLabel preprocessor using pos tag.'

    def combine_pos_tag(self, pos_tag):
        noun = ['NN', 'NNS', 'NNP', 'NNPS']
        adjective = ['JJ', 'JJR', 'JJS']
        adverb = ['RB', 'RBR', 'RBS']
        verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        wh = ['WDT', 'WP', 'WRB']
        if pos_tag in noun:
            return 'NN'
        elif pos_tag in adjective:
            return 'JJ'
        elif pos_tag in adverb:
            return 'RB'
        elif pos_tag in verb:
            return 'VB'
        elif pos_tag in wh:
            return 'WP'
        else:
            return pos_tag

    def prelabel_with_pos(self, former_labels=['null']):
        fopen_train = open(self.train_file)
        lines_train = fopen_train.readlines()
        fopen_train.close()
        fopen_test = open(self.test_file)
        lines_test = fopen_test.readlines()
        fopen_test.close()
        with open(self.train_file, 'w') as fopen:
            for line in lines_train:
                if line.strip():
                    token = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[1]
                    if label.lower() in former_labels:
                        label = nltk.pos_tag([token])
                        fopen.write(token + ' ' + self.combine_pos_tag(label[0][1]) + '\n')
                    else:
                        fopen.write(line)
                else:
                    fopen.write(line)
        with open(self.test_file, 'w') as fopen:
            for line in lines_test:
                if line.strip():
                    token = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[1]
                    if label.lower() in former_labels:
                        label = nltk.pos_tag([token])
                        fopen.write(token + ' ' + self.combine_pos_tag(label[0][1]) + '\n')
                    else:
                        fopen.write(line)
                else:
                    fopen.write(line)

    def prelabel_with_pos_by_sentence(self, former_labels=['null']):
        fopen_train = open(self.train_file)
        lines_train = fopen_train.readlines()
        fopen_train.close()
        fopen_test = open(self.test_file)
        lines_test = fopen_test.readlines()
        fopen_test.close()
        with open(self.train_file, 'w') as fopen:
            tokens, labels = [], []
            for line in lines_train:
                if line.strip():
                    token = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[1]
                    tokens.append(token)
                    labels.append(label)
                else:
                    pos_tags = nltk.pos_tag(tokens)
                    for (token, pos_tag), label in zip(pos_tags, labels):
                        if label.lower() in former_labels:
                            fopen.write(token + ' ' + self.combine_pos_tag(pos_tag) + '\n')
                        else:
                            fopen.write(token + ' ' + label + '\n')
                    fopen.write(line)
                    tokens, labels = [], []
        with open(self.test_file, 'w') as fopen:
            tokens, labels = [], []
            for line in lines_test:
                if line.strip():
                    token = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[1]
                    tokens.append(token)
                    labels.append(label)
                else:
                    pos_tags = nltk.pos_tag(tokens)
                    for (token, pos_tag), label in zip(pos_tags, labels):
                        if label.lower() in former_labels:
                            fopen.write(token + ' ' + self.combine_pos_tag(pos_tag) + '\n')
                        else:
                            fopen.write(token + ' ' + label + '\n')
                    fopen.write(line)
                    tokens, labels = [], []

    def prelabel_with_pos_nocombined_by_sentence(self, former_labels=['null']):
        fopen_train = open(self.train_file)
        lines_train = fopen_train.readlines()
        fopen_train.close()
        fopen_test = open(self.test_file)
        lines_test = fopen_test.readlines()
        fopen_test.close()
        with open(self.train_file, 'w') as fopen:
            tokens, labels = [], []
            for line in lines_train:
                if line.strip():
                    token = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[1]
                    tokens.append(token)
                    labels.append(label)
                else:
                    pos_tags = nltk.pos_tag(tokens)
                    for (token, pos_tag), label in zip(pos_tags, labels):
                        if label.lower() in former_labels:
                            fopen.write(token + ' ' + pos_tag + '\n')
                        else:
                            fopen.write(token + ' ' + label + '\n')
                    fopen.write(line)
                    tokens, labels = [], []
        with open(self.test_file, 'w') as fopen:
            tokens, labels = [], []
            for line in lines_test:
                if line.strip():
                    token = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[1]
                    tokens.append(token)
                    labels.append(label)
                else:
                    pos_tags = nltk.pos_tag(tokens)
                    for (token, pos_tag), label in zip(pos_tags, labels):
                        if label.lower() in former_labels:
                            fopen.write(token + ' ' + pos_tag + '\n')
                        else:
                            fopen.write(token + ' ' + label + '\n')
                    fopen.write(line)
                    tokens, labels = [], []


class LSTMPreprocessor(Preprocessor):
    def preprocess(self):
        print 'This is LSTM preprocessor.'

    def get_train_data(self):
        pass
