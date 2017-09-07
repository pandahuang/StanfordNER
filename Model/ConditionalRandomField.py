from nltk.internals import java
from subprocess import PIPE
from datetime import datetime
import time
from Data.DataManager import DataManager
import shlex
import os


def CombineTokens(tokens):
    strg = reduce(lambda x, y: str(x) + ' ' + str(y), tokens)
    return strg


def CountTrainResults(sout_train, serr_train):
    sout, serr = sout_train, serr_train
    results = serr.strip().split('\n')
    train_datasize, train_time = 0, 0
    for result in results:
        try:
            infos = result.strip().split(':')
            if infos[0].strip() == 'numDocuments':
                train_datasize = infos[1].strip()
        except IndexError:
            continue
    train_time_res = results[-3]
    train_time = train_time_res.strip().split('[')[1].strip().split(' ')[0]
    return train_datasize, train_time


def CountTestResults(sout_test, serr_test):
    sout, serr = sout_test, serr_test
    labels_tp, labels_fp, labels_fn = {}, {}, {}
    results_sout = sout.strip().split('\r')
    count_wrong, count_sent = 0, 0
    isWorng = False
    for result in results_sout:
        if result.strip():
            token, label, res = result.split('\t')[0], result.split('\t')[1], result.split('\t')[2]
            if label != res:
                isWorng = True
        else:
            if isWorng:
                count_wrong += 1
            count_sent += 1
            isWorng = False
        try:
            token, label, res = result.split('\t')[0], result.split('\t')[1], result.split('\t')[2]
            if label == res:
                labels_tp[label] = labels_tp[label] + 1 if labels_tp.has_key(label) else 1
            else:
                labels_fn[label] = labels_fn[label] + 1 if labels_fn.has_key(label) else 1
                labels_fp[res] = labels_fp[res] + 1 if labels_fp.has_key(res) else 1
        except IndexError, e1:
            continue
        except KeyError, e2:
            continue

    sent_accuracy = float(count_sent - count_wrong) / count_sent

    print '--------------------Custom Result--------------------'
    print 'The sentences accuracy is %f.' % (sent_accuracy)
    # print 'Entity\tP\tR\tF1\tTP\tFP\tFN'
    detail_result = 'Entity\tP\tR\tF1\tTP\tFP\tFN'
    labelset = list(set(labels_fp.keys() + labels_tp.keys() + labels_fn.keys()))
    for label in labelset:
        try:
            tp = labels_tp[label] if labels_tp.has_key(label) else 0
            fp = labels_fp[label] if labels_fp.has_key(label) else 0
            fn = labels_fn[label] if labels_fn.has_key(label) else 0
            precision = float(tp) / (tp + fp)
            recall = float(tp) / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            detail_result = detail_result + '\n' + '%s\t%f\t%f\t%f\t%d\t%d\t%d' % (
                label, precision, recall, f1, tp, fp, fn)
        except ZeroDivisionError, e:
            continue
    total_tp = sum(labels_tp.values())
    total_fp = sum(labels_fp.values())
    total_fn = sum(labels_fn.values())
    total_precision = float(total_tp) / (total_tp + total_fp)
    total_recall = float(total_tp) / (total_tp + total_fn)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    detail_result = detail_result + '\n' + '%s\t%f\t%f\t%f\t%d\t%d\t%d' % (
        'Total', total_precision, total_recall, total_f1, total_tp, total_fp, total_fn)
    test_datasize, test_time = 0, 0
    results_serr = serr.strip().split('\n')
    for result in results_serr:
        try:
            infos = result.strip().split(' ')
            if infos[0].strip() == 'CRFClassifier':
                test_datasize = infos[5].strip()
                test_time = int(infos[2].strip()) * (1 / float(infos[8].strip()))
        except IndexError:
            continue
    return sent_accuracy, test_datasize, test_time, detail_result


def LogResult(exp_date, sent_accuracy, source_data, train_datasize, train_time, test_datasize, test_time,
              result_file='result-record.txt'):
    if not os.path.exists(result_file):
        fopen = open(result_file, 'w')
        fopen.write('ExpDate' + '\t' + 'SentAccuracy' + '\t' + 'SourceData' + '\t' + 'TrainDataSize' + '\t' +
                    'TrainTime' + '\t' + 'TestDataSize' + '\t' + 'TestTime' + '\n')
        fopen.close()
    fopen = open(result_file, 'a')
    fopen.write(exp_date + '\t' + sent_accuracy + '\t' + source_data + '\t' + str(train_datasize) + '\t' + str(train_time) + '\t' +
                str(test_datasize) + '\t' + str(test_time) + '\n')
    fopen.close()


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

    def train_temp(self):
        command = self.command_line + ' -prop ' + self.prop_file
        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        print '--------------------TRAIN------------------------'
        print serr
        return sout, serr

    def training_crf_model(self):
        command = self.command_line + ' -prop ' + self.prop_file
        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        train_datasize, train_time = CountTrainResults(sout, serr)
        print '--------------------TRAIN------------------------'
        print serr
        print 'Training time is %s seconds.' % (train_time)
        return sout, serr, train_datasize, train_time

    def train(self):
        return self.training_crf_model()

    def verify_by_sentence(self, sentence):
        pass

    def verify_temp(self, file=''):
        if not file:
            file = self.test_file
        command = self.command_line + ' -loadClassifier ' + self.model_filename + ' -testFile ' + file
        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        print '--------------------TEST------------------------'
        print serr
        return sout, serr


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

        cmd = shlex.split(command)
        sout, serr = java(cmd, classpath=self.path_to_jar, stdout=PIPE, stderr=PIPE)
        sent_accuracy, test_datasize, test_time, detail_result = CountTestResults(sout, serr)
        print '--------------------TEST------------------------'
        print serr
        print 'Testing time is %s seconds.' % (test_time)
        if not test_file == self.test_file:
            os.remove(test_file)
        return sout, serr, sent_accuracy, test_datasize, test_time, detail_result

    def train_and_verify_temp(self, train_file='', test_file=''):
        if not train_file: train_file = self.train_file
        if not test_file: test_file = self.test_file
        sout_train, serr_train = self.train(train_file)
        sout_test, serr_test = self.verify(test_file)
        return sout_train, serr_train, sout_test, serr_test

    def train_and_verify(self):
        sout_train, serr_train, train_datasize, train_time = self.train()
        sout_test, serr_test, sent_accuracy, test_datasize, test_time, detail_result = self.verify()
        exp_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_data = self.source_data_file
        LogResult(exp_date, str(sent_accuracy), source_data, train_datasize, train_time, test_datasize, test_time)
        return sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result


if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   train_file=DM.train_file, test_file=DM.test_file)
    crf_test.feature_config()
    crf_test.train()
    crf_test.verify()
