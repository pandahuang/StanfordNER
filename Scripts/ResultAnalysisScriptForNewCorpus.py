import os
import copy
from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from Visualization import PainterFactory

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

feature_sets = []


def list2dict(feature):
    return {key: features.get(key) for key in feature if features.has_key(key)}


DM = DataManager()
DM.change_pwd()
DM.source_data_file = 'CorpusLabelData_MergedFilter.txt'
DM.remove('LogWrongSents.txt')


def run(feature_set, DM=DM):
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=True)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                   result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result = crf_test.train_and_verify()
    DM.train_file = 'CorpusLabelData_MergedFilter_Full.txt'
    crf_test_f = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                     source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                     result_file=DM.result_file)
    crf_test_f.feature_config(features=feature_set)
    sout_train_f, serr_train_f, sent_accuracy_f, sout_test_f, serr_test_f, detail_result_f = crf_test_f.train_and_verify()
    return sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result, sout_train_f, serr_train_f, sent_accuracy_f, sout_test_f, serr_test_f, detail_result_f


def ResultsAndWrongAnswerRecord(sout, serr, detail_result):
    results = sout.strip().split('\r')
    isWorng = False
    sents = []
    with open('LogWrongSents.txt', 'a') as fopen:
        fopen.write(detail_result + '\n')
        fopen.write('----------------------------------------------------------------\n')
    for result in results:
        if result.strip():
            sents.append(result.strip())
            token, label, res = result.split('\t')[0], result.split('\t')[1], result.split('\t')[2]
            if label != res:
                isWorng = True
        else:
            with open('LogWrongSents.txt', 'a') as fopen:
                if sents and isWorng:
                    for sent in sents:
                        fopen.write(sent + '\n')
                    fopen.write('\n')
            sents = []
            isWorng = False
    with open('LogWrongSents.txt', 'a') as fopen:
        fopen.write('===============================================================================' + '\n')


if __name__ == '__main__':
    cycle_times = 1
    sent_accuracys, sent_accuracys_f = [], []
    for i in range(cycle_times):
        # use demo features
        feature_demo = features
        sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result, sout_train_f, serr_train_f, sent_accuracy_f, sout_test_f, serr_test_f, detail_result_f = run(
            feature_demo)
        ResultsAndWrongAnswerRecord(sout_test, serr_test, detail_result)
        ResultsAndWrongAnswerRecord(sout_test_f, serr_test_f, detail_result_f)
        sent_accuracys.append(sent_accuracy)
        sent_accuracys_f.append(sent_accuracy_f)
    print 'Average sent_accuracy of former corpus is : %f' % (sum(sent_accuracys) / cycle_times)
    print 'Average sent_accuracy of new corpus is : %f' % (sum(sent_accuracys_f) / cycle_times)
