import random
from Preprocessing.Datum import Datum


def CombineTokens(tokens):
    strg = reduce(lambda x, y: str(x) + ' ' + str(y), tokens)
    return strg


def regular_sample(index_record, percent):
    train_index = []
    test_index = []
    slice_len = percent[0] + percent[1]
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

    def get_train_data(self, datums, percent=(2, 1), isRandom=True):
        datum_index = range(len(datums))
        train_index, test_index = [], []
        if percent == (0, 0):
            train_index = test_index = datum_index
        else:
            if isRandom:
                train_index = random.sample(datum_index, len(datum_index) / (percent[0] + percent[1]) * percent[0])
                test_index = list(set(datum_index) - set(train_index))
            else:
                train_index, test_index = regular_sample(datum_index, percent)
        with open(self.train_file, 'w') as fopen_train, open(self.test_file, 'w') as fopen_test:
            for index, datum in enumerate(datums):
                if index in train_index:
                    for token, glabel in zip(datum.tokens, datum.golden_labels):
                        fopen_train.write(token + '\t' + glabel + '\n')
                    fopen_train.write('\n')
                if index in test_index:
                    for token, glabel in zip(datum.tokens, datum.golden_labels):
                        fopen_test.write(token + '\t' + glabel + '\n')
                    fopen_test.write('\n')
        return train_index, test_index


class LSTMPreprocessor(Preprocessor):
    def preprocess(self):
        print 'This is LSTM preprocessor.'

    def get_train_data(self):
        pass
