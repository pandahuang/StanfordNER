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
    'printFeatures': '1'
}

feature_sets = []


def list2dict(feature):
    return {key: features.get(key) for key in feature if features.has_key(key)}


default = ['useWord', 'useSequences', 'printFeatures']
word = ['usePrev', 'useNext']
ngram = ['useNGrams', 'noMidNGrams', 'maxNGramLeng']
wordshape = ['wordShape', 'useTypeSeqs', 'useTypeSeqs2', 'useTypeySequences']
classf = ['useClassFeature', 'usePrevSequences', 'maxLeft']
disjunctive = ['useDisjunctive']
lemmas = ['useLemmas', 'usePrevNextLemmas', 'useLemmaAsWord']
position = ['usePosition', 'useBeginSent']

feature_default = list2dict(default)
feature_sets.append(feature_default)
feature_word = list2dict(default + word)
feature_sets.append(feature_word)
feature_ngram = list2dict(default + ngram)
feature_sets.append(feature_ngram)
feature_shape = list2dict(default + wordshape)
feature_sets.append(feature_shape)
feature_class = list2dict(default + classf)
feature_sets.append(feature_class)
feature_disjunctive = list2dict(default + disjunctive)
feature_sets.append(feature_disjunctive)

DM = DataManager()
DM.change_pwd()


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
    sout, serr = run(feature_demo)
    results = sout.strip().split('\r')
    isWorng = False
    sents = []
    validation = serr.strip().split('\r')[-15:]
    with open('LogWrongSents.txt', 'a') as fopen:
        for val in validation:
            fopen.write(val)
        fopen.write('\n')
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
