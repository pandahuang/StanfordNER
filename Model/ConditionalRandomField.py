from nltk.internals import java
from subprocess import PIPE
from Data.DataManager import DataManager
import shlex


def CombineTokens(tokens):
    strg = reduce(lambda x, y: str(x) + ' ' + str(y), tokens)
    return strg


class CRF(object):
    command_line = 'edu.stanford.nlp.ie.crf.CRFClassifier'

    def __init__(self, **kw):
        self.path_to_jar = kw.get('path_to_jar')
        self.prop_file = kw.get('prop_file')
        self.model_filename = kw.get('model_file')  # path of custom model
        self.source_data_file = kw.get('source_data_file')
        self.train_file = kw.get('train_file')
        self.test_file = kw.get('test_file')
        self.result_file = kw.get('result_file')

    def feature_config(self, features={}):
        trainFile = 'trainFile' + '=' + self.train_file
        serializeTo = 'serializeTo' + '=' + self.model_filename
        structure = 'map = word=0,answer=1'
        with open(self.prop_file, 'w') as fopen:
            fopen.write(trainFile + '\n')
            fopen.write(serializeTo + '\n')
            fopen.write(structure + '\n')
            for fn, fv in features.iteritems():
                fopen.write(fn + '=' + fv + '\n')
        pass

    def train(self, file=''):
        if not file:
            file = self.train_file
        command = self.command_line + ' -prop ' + self.prop_file
        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        print '--------------------TRAIN------------------------'
        print serr
        return sout, serr

    def verify_by_sentence(self, sentence):
        pass

    def verify(self, file=''):
        if not file:
            file = self.test_file
        command = self.command_line + ' -loadClassifier ' + self.model_filename + ' -testFile ' + file
        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        print '--------------------TEST------------------------'
        print serr
        return sout, serr

    def train_and_verify(self, train_file='', test_file=''):
        if not train_file: train_file = self.train_file
        if not test_file: test_file = self.test_file
        sout_train, serr_train = self.train(train_file)
        sout_test, serr_test = self.verify(test_file)
        return sout_train, serr_train, sout_test, serr_test


if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   train_file=DM.train_file, test_file=DM.test_file)
    crf_test.feature_config()
    crf_test.train()
    crf_test.verify()
