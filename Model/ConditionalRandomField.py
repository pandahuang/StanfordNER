from nltk.internals import java
from subprocess import PIPE
from datetime import datetime
import time
from Data.DataManager import DataManager
import shlex
import os


def CombineTokens(token_list):
    strg = reduce(lambda x, y: str(x) + ' ' + str(y), token_list)
    return strg


def CountCustomResults(sout, serr):
    custom_info = ''
    labels_tp, labels_fp, labels_fn = {}, {}, {}
    results = sout.strip().split('\r')
    count = 0
    for result in results:
        try:
            token, label, res = result.split('\t')[0], result.split('\t')[1], result.split('\t')[2]
            if label == res:
                labels_tp[label] = labels_tp[label] + 1 if labels_tp.has_key(label) else 1
                count += 1
            else:
                labels_fn[label] = labels_fn[label] + 1 if labels_fn.has_key(label) else 1
                labels_fp[res] = labels_fp[res] + 1 if labels_fp.has_key(res) else 1
        except IndexError, e1:
            # print result
            continue
        except KeyError, e2:
            continue
    accuracy = float(count) / len(results)
    print '--------------------Custom Result--------------------'
    print 'Total accuracy is %f.' % (accuracy)
    print 'Entity\tP\tR\tF1\tTP\tFP\tFN'
    custom_info = 'Entity\tP\tR\tF1\tTP\tFP\tFN'
    labelset = list(set(labels_fp.keys() + labels_tp.keys() + labels_fn.keys()))
    for label in labelset:
        try:
            tp = labels_tp[label] if labels_tp.has_key(label) else 0
            fp = labels_fp[label] if labels_fp.has_key(label) else 0
            fn = labels_fn[label] if labels_fn.has_key(label) else 0
            precision = float(tp) / (tp + fp)
            recall = float(tp) / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            print '%s\t%f\t%f\t%f\t%d\t%d\t%d' % (label, precision, recall, f1, tp, fp, fn)
            custom_info = custom_info + '\n' + '%s\t%f\t%f\t%f\t%d\t%d\t%d' % (label, precision, recall, f1, tp, fp, fn)
        except ZeroDivisionError, e:
            continue
    total_tp = sum(labels_tp.values())
    total_fp = sum(labels_fp.values())
    total_fn = sum(labels_fn.values())
    total_precision = float(total_tp) / (total_tp + total_fp)
    total_recall = float(total_tp) / (total_tp + total_fn)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    print '%s\t%f\t%f\t%f\t%d\t%d\t%d' % (
        'Total', total_precision, total_recall, total_f1, total_tp, total_fp, total_fn)
    custom_info = custom_info + '\n' + '%s\t%f\t%f\t%f\t%d\t%d\t%d' % (
        'Total', total_precision, total_recall, total_f1, total_tp, total_fp, total_fn)
    print '--------------------System Result--------------------'
    print serr
    return accuracy, custom_info


def LogResult(time, accuracy, serr, result_file='result-questions.txt'):
    totals = serr.strip().split('\r')[-1].split('\t')
    totals_precision = totals[1]
    totals_recall = totals[2]
    totals_f1 = totals[3]
    if not os.path.exists(result_file):
        fopen = open(result_file, 'w')
        fopen.write('time' + '\t' + 'accuracy' + '\t' + 'precision' + '\t' + 'recall' + '\t' + 'f-value' + '\n')
        fopen.close()
    fopen = open(result_file, 'a')
    fopen.write(time + '\t' + accuracy + '\t' + totals_precision + '\t' + totals_recall + '\t' + totals_f1 + '\n')
    fopen.close()
    pass


class CRF(object):
    command_line = 'edu.stanford.nlp.ie.crf.CRFClassifier'

    def __init__(self, **kw):
        self.path_to_jar = kw.get('path_to_jar')
        self.prop_file = kw.get('prop_file')
        self.model_filename = kw.get('model_file')  # path of custom model
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

    def training_crf_model(self):
        before_training = datetime.now()
        command = self.command_line + ' -prop ' + self.prop_file
        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        print '--------------------TRAIN------------------------'
        print sout
        print serr
        after_training = datetime.now()
        training_time = after_training - before_training
        print 'Training time is %s.%s seconds.' % (training_time.seconds, training_time.microseconds)

    def train(self):
        # if not os.path.exists(self.model_filename):
        self.training_crf_model()

    def verify(self, sentence=None):
        test_file = self.test_file
        sents = []
        if sentence:
            sents.extend(sentence)
            test_file = 'temp-test-samples.txt'
            with open(test_file, 'w') as fopen:
                for sent in sents:
                    for token in sent.strip().split():
                        fopen.write(token + ' Unknown' + '\n')
                    fopen.write('\n')

        command = self.command_line + ' -loadClassifier ' + self.model_filename + ' -testFile ' + test_file

        print '--------------------TEST------------------------'
        before_testing = datetime.now()
        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        accuracy, custom_info = CountCustomResults(sout, serr)
        after_testing = datetime.now()
        testing_time = after_testing - before_testing
        print 'Testing time is %s.%s seconds.' % (testing_time.seconds, testing_time.microseconds)
        LogResult(str(testing_time.total_seconds()), str(accuracy), serr)
        if not test_file == self.test_file:
            os.remove(test_file)
        return sout, serr, custom_info


if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   train_file=DM.train_file, test_file=DM.test_file)
    crf_test.feature_config()
    crf_test.train()
    crf_test.verify()
