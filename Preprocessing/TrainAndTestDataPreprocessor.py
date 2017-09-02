import random
from Preprocessor import Preprocessor


class TrainAndTestDataPreprocessor(Preprocessor):
    def __init__(self, **kw):
        self.source_data_file = kw.get('source_data_file')
        self.train_file = kw.get('train_file')
        self.test_file = kw.get('test_file')

    def preprocess(self):
        print 'This is a preprocessor deal with train data and test data.'

    def reduce_replicate_data(self):
        fopen_train = open(self.train_file)
        lines_train = fopen_train.readlines()
        fopen_train.close()
        fopen_test = open(self.test_file)
        lines_test = fopen_test.readlines()
        fopen_test.close()
        sentence_train = []
        sentence_test = []
        temp_sentence = ''
        for line in lines_train:
            if line.strip():
                temp_sentence = temp_sentence + line
            else:
                sentence_train.append(temp_sentence)
                temp_sentence = ''
        temp_sentence = ''
        for line in lines_test:
            if line.strip():
                temp_sentence = temp_sentence + line
            else:
                sentence_test.append(temp_sentence)
                temp_sentence = ''
        reduced_sentence_train = list(set(sentence_train) - set(sentence_test))
        with open(self.train_file, 'w') as fopen:
            for sent in reduced_sentence_train:
                fopen.write(sent)
                fopen.write('\n')
