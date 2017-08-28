import os
import copy
from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from Visualization import CanvasFactory

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
    'useLemmas': 'true',
    'usePrevNextLemmas': 'true',
    'useLemmaAsWord': 'true',
    'usePosition': 'true',
    'useBeginSent': 'true',
    'printFeatures': '1',
    'mergeTags': 'false'
}

feature_sets = []


def list2dict(feature):
    return {key: features.get(key) for key in feature if features.has_key(key)}


DM = DataManager()
DM.change_pwd()
DM.source_data_file = 'CorpusLabelData_MergedFilter.txt'


def run(feature_set, DM=DM):
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=True)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   train_file=DM.train_file,
                   test_file=DM.test_file, result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    crf_test.train()
    if os.path.exists(os.path.join(os.getcwd(), 'features-train.txt')):
        os.remove(os.path.join(os.getcwd(), 'features-train.txt'))
    if os.path.exists(os.path.join(os.getcwd(), 'features-test.txt')):
        os.remove(os.path.join(os.getcwd(), 'features-test.txt'))
    if os.path.exists(os.path.join(os.getcwd(), 'LogWrongSents.txt')):
        os.remove(os.path.join(os.getcwd(), 'LogWrongSents.txt'))
    os.rename(os.path.join(os.getcwd(), 'features-1.txt'), os.path.join(os.getcwd(), 'features-train.txt'))
    sout, serr, sent_accuracy, custom_info = crf_test.verify()
    os.rename(os.path.join(os.getcwd(), 'features-1.txt'), os.path.join(os.getcwd(), 'features-test.txt'))
    return sout, serr, sent_accuracy, custom_info


for i in range(1):
    # use demo features
    feature_demo = features
    sout, serr, sent_accuracy, custom_info = run(feature_demo)
    results = sout.strip().split('\r')
    isWorng = False
    sents = []
    with open('LogWrongSents.txt', 'a') as fopen:
        fopen.write(custom_info + '\n')
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

import numpy as np

field_res, null_res, total_res = [], [], []
with open('LogWrongSents.txt') as fopen:
    for line in fopen:
        if line:
            if line.strip().split('\t')[0] in ['B-Field', 'I-Field'] and len(line.strip().split('\t')) == 7:
                field_res.append(line.strip().split('\t')[1:])  # P, R, F1, TP, FP, FN
            elif line.strip().split('\t')[0] == 'NULL' and len(line.strip().split('\t')) == 7:
                null_res.append(line.strip().split('\t')[1:])
            elif line.strip().split('\t')[0] == 'Total' and len(line.strip().split('\t')) == 7:
                total_res.append(line.strip().split('\t')[1:])
field_res = np.array(field_res).astype(np.float)
null_res = np.array(null_res).astype(np.float)
total_res = np.array(total_res).astype(np.float)

# print field_res

field_res_avg = [float(np.mean(field_res[:, i])) for i in range(6)]
null_res_avg = [float(np.mean(null_res[:, i])) for i in range(6)]
total_res_avg = [float(np.mean(total_res[:, i])) for i in range(6)]
print ' '.join([str(t) for t in total_res_avg])