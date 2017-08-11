import random


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
            train_index = random.sample(index_record, len(index_record) / percent[0] * (percent[0] + percent[1]))
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


class LSTMPreprocessor(Preprocessor):
    def preprocess(self):
        print 'This is LSTM preprocessor.'

    def get_train_data(self):
        pass
