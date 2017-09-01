import os
import copy
from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF

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
DM.source_data_file = 'CorpusLabelData_MergedFilter_Update.txt'
DM.remove('LogWrongSents.txt')


def run(feature_set, DM=DM):
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=True)
    prepos_processor = ProcessorFactory.PreLabelWithPosProcessorFactory().produce(train_file=DM.train_file,
                                                                                  test_file=DM.test_file)
    prepos_processor.prelabel_with_pos()
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                   result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result = crf_test.train_and_verify()
    return sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result


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
    sent_accuracys = []
    for i in range(1):
        # use demo features
        feature_demo = features
        sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result = run(feature_demo)
        ResultsAndWrongAnswerRecord(sout_test, serr_test, detail_result)
        sent_accuracys.append(sent_accuracy)
    print 'Average sent_accuracy is : %f' % (sum(sent_accuracys) / 10)
