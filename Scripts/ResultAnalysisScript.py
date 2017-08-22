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
DM.source_data_file = 'CorpusLabelData_WithToken.txt'


def run(feature_set, DM=DM):
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=True)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   train_file=DM.train_file,
                   test_file=DM.test_file, result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    crf_test.train()
    return crf_test.verify()


for i in range(10):
    # use demo features
    feature_demo = features
    sout, serr, custom_info = run(feature_demo)
    results = sout.strip().split('\r')
    isWorng = False
    sents = []
    validation = serr.strip().split('\r')[-15:]
    with open('LogWrongSents.txt', 'a') as fopen:
        # for val in validation:
        #     fopen.write(val)
        # fopen.write('\n')
        # fopen.write('----------------------------------------------------------------\n')
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

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(9, 6))
# xticks = ['Precision', 'Recall', 'F1']
# plt.xticks(range(len(xticks)), xticks)
# plt.ylim(0.5, 1)
# # plt.bar(np.arange(len(xticks)), field_res_avg[:2], width=0.2, facecolor='lightskyblue', edgecolor='white',
# #         label='Field')
# # plt.legend(loc='upper left', frameon=False)
# # for x, y in zip(np.arange(len(xticks)), field_res_avg[:2]):
# #     plt.text(x, y / 2, '%.2f' % y, ha='center')
# # plt.bar(np.arange(len(xticks)), null_res_avg[:2], width=0.2, facecolor='lightgreen', edgecolor='white', label='NULL',
# #         bottom=field_res_avg[:2])
# # plt.legend(loc='upper left', frameon=False)
# # for x, y, y1 in zip(np.arange(len(xticks)), np.array(null_res_avg[:2]), np.array(field_res_avg[:2])):
# #     plt.text(x, y / 2 + y1, '%.2f' % y, ha='center')
# # plt.bar(np.arange(len(xticks)), np.array(total_res_avg[:2]) - np.array(field_res_avg[:2]) - np.array(null_res_avg[:2]),
# #         width=0.2, facecolor='lightpink', edgecolor='white', label='Others',
# #         bottom=(np.array(field_res_avg[:2]) + np.array(null_res_avg[:2])))
# # plt.legend(loc='upper left', frameon=False)
# # for x, y, y1, y2 in zip(np.arange(len(xticks)),
# #                         np.array(total_res_avg[:2]) - np.array(field_res_avg[:2]) - np.array(null_res_avg[:2]),
# #                         np.array(null_res_avg[:2]), np.array(field_res_avg[:2])):
# #     plt.text(x, y / 2 + y1 + y2, '%.2f' % y, ha='center')
# # plt.show()
#
# plt.bar(np.arange(len(xticks)), field_res_avg[:3], width=0.2, facecolor='lightskyblue', edgecolor='white',
#         label='Field')
# plt.legend(loc='upper left', frameon=False)
# for x, y in zip(np.arange(len(xticks)), field_res_avg[:3]):
#     plt.text(x, y + 0.02, '%.2f' % y, ha='center')
#
# plt.bar(np.arange(len(xticks)) + 3, null_res_avg[:3], width=0.2, facecolor='lightgreen', edgecolor='white',
#         label='NULL')
# plt.legend(loc='upper left', frameon=False)
# for x, y in zip(np.arange(len(xticks)) + 3, null_res_avg[:3]):
#     plt.text(x, y + 0.02, '%.2f' % y, ha='center')
#
# plt.bar(np.arange(len(xticks)) + 6, total_res_avg[:3], width=0.2, facecolor='lightpink', edgecolor='white',
#         label='Total')
# plt.legend(loc='upper left', frameon=False)
# for x, y in zip(np.arange(len(xticks)) + 6, total_res_avg[:3]):
#     plt.text(x, y + 0.02, '%.2f' % y, ha='center')
# plt.show()
