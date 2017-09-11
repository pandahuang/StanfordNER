import os
from datetime import datetime
import time


class ScriptToolkit(object):
    def __init__(self, DM):
        self.DM = DM

    @classmethod
    def get_demo_features(cls):
        features = {
            'useClassFeature': 'true',
            'useWord': 'true',
            'useNGrams': 'true',
            'noMidNGrams': 'true',
            'useDisjunctive': 'true',
            'maxNGramLeng': '6',
            'usePrev': 'true',
            'useNext': 'true',
            'useSequences': 'true',
            'usePrevSequences': 'true',
            'maxLeft': '1',
            'useTypeSeqs': 'true',
            'useTypeSeqs2': 'true',
            'useTypeySequences': 'true',
            'wordShape': 'chris2useLC',
        }
        return features

    def list2dict(self, features, feature):
        return {key: features.get(key) for key in feature if features.has_key(key)}

    def LogResultsAndWrongAnswer(self, sout, serr, detail_result):
        results = sout.strip().split('\r')
        isWorng = False
        sents = []
        with open(self.DM.log_wrong_sentences, 'a') as fopen:
            fopen.write(detail_result + '\n')
            fopen.write('----------------------------------------------------------------\n')
        for result in results:
            if result.strip():
                sents.append(result.strip())
                token, label, res = result.split('\t')[0], result.split('\t')[1], result.split('\t')[2]
                if label != res:
                    isWorng = True
            else:
                with open(self.DM.log_wrong_sentences, 'a') as fopen:
                    if sents and isWorng:
                        for sent in sents:
                            fopen.write(sent + '\n')
                        fopen.write('\n')
                sents = []
                isWorng = False
        with open(self.DM.log_wrong_sentences, 'a') as fopen:
            fopen.write('===============================================================================' + '\n')

    def ReadBestAndWorstDataset(self):
        with open(self.DM.train_file) as fopen:
            lines_train = fopen.readlines()
        with open(self.DM.test_file) as fopen:
            lines_test = fopen.readlines()
        return (lines_train, lines_test)

    def WriteBestAndWorstDataset(self, max_accuracy, max_data, min_accuracy, min_data):
        with open(self.DM.log_best_dataset, 'w') as fopen:
            fopen.write(str(max_accuracy) + '\n')
            fopen.write('Train--------------------------------------------------------------------------' + '\n')
            for line in max_data[0]:
                fopen.write(line)
            fopen.write('Test--------------------------------------------------------------------------' + '\n')
            for line in max_data[1]:
                fopen.write(line)
        with open(self.DM.log_worst_dataset, 'w') as fopen:
            fopen.write(str(min_accuracy) + '\n')
            fopen.write('Train--------------------------------------------------------------------------' + '\n')
            for line in min_data[0]:
                fopen.write(line)
            fopen.write('Test--------------------------------------------------------------------------' + '\n')
            for line in min_data[1]:
                fopen.write(line)

    @classmethod
    def StatisticDatums(cls, datums):
        sentences_amount, tokens_distribution, glabels_distribution = 0, {}, {}
        sentences_amount = len(datums)
        for datum in datums:
            for token in datum.tokens:
                if tokens_distribution.has_key(token):
                    tokens_distribution[token] += 1
                else:
                    tokens_distribution[token] = 1
        for datum in datums:
            for glabel in datum.golden_labels:
                if glabels_distribution.has_key(glabel):
                    glabels_distribution[glabel] += 1
                else:
                    glabels_distribution[glabel] = 1
        return sentences_amount, sorted(tokens_distribution.iteritems(), key=lambda d: d[1], reverse=True), \
               sorted(glabels_distribution.iteritems(), key=lambda d: d[1], reverse=True)

    @classmethod
    def ParseTrainSoutAndSerr(cls, sout, serr):
        if sout.strip():
            pass
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

    @classmethod
    def ParseTestSoutAndSerr(cls, sout, serr):
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
        return sent_accuracy, test_datasize, round(test_time, 2), detail_result

    @classmethod
    def LogResult(cls, sent_accuracy, source_data, train_datasize, train_time, test_datasize, test_time,
                  result_file='result-record.txt'):
        if not os.path.exists(result_file):
            fopen = open(result_file, 'w')
            fopen.write('ExpDate' + '\t' + 'SentAccuracy' + '\t' + 'SourceData' + '\t' + 'TrainDataSize' + '\t' +
                        'TrainTime' + '\t' + 'TestDataSize' + '\t' + 'TestTime' + '\n')
            fopen.close()
        fopen = open(result_file, 'a')
        exp_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fopen.write(exp_date + '\t' + str(sent_accuracy) + '\t' + source_data + '\t' + str(train_datasize) + '\t' + str(train_time) +
                    '\t' + str(test_datasize) + '\t' + str(test_time) + '\n')
        fopen.close()
