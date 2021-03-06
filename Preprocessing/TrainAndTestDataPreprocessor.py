import random
from Preprocessor import Preprocessor


class TrainAndTestDataPreprocessor(Preprocessor):
    def __init__(self, **kw):
        self.source_data_file = kw.get('source_data_file')
        self.train_file = kw.get('train_file')
        self.test_file = kw.get('test_file')

    def preprocess(self):
        print 'This is a preprocessor deal with train data and test data.'

    def reduce_replicate_data(self, first_datums, second_datums, output_file=''):
        if output_file == '':
            output_file = self.train_file
        second_datums_sentences = [datum.get_sentence() for datum in second_datums]
        with open(output_file, 'w') as fopen:
            for datum in first_datums:
                if not datum.get_sentence() in second_datums_sentences:
                    for token, glabel in zip(datum.tokens, datum.golden_labels):
                        fopen.write(token + '\t' + glabel + '\n')
                    fopen.write('\n')

    def reduce_replicate_data(self, first_file='', second_file='', third_file=''):
        if first_file == '':
            first_file = self.source_data_file
        if second_file == '':
            second_file = self.test_file
        if third_file == '':
            third_file = self.train_file
        fopen_first = open(first_file)
        lines_first = fopen_first.readlines()
        fopen_first.close()
        fopen_second = open(second_file)
        lines_second = fopen_second.readlines()
        fopen_second.close()
        sentence_first = []
        sentence_second = []
        temp_sentence = ''
        for line in lines_first:
            line = line.replace(' ', '\t')
            if line.strip():
                temp_sentence = temp_sentence + line
            else:
                sentence_first.append(temp_sentence)
                temp_sentence = ''
        temp_sentence = ''
        for line in lines_second:
            line = line.replace(' ', '\t')
            if line.strip():
                temp_sentence = temp_sentence + line
            else:
                sentence_second.append(temp_sentence)
                temp_sentence = ''
        reduced_sentence = list(set(sentence_first) - set(sentence_second))
        with open(third_file, 'w') as fopen:
            for sent in reduced_sentence:
                fopen.write(sent)
                fopen.write('\n')
