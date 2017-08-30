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

# feature_word_ngram = list2dict(default + word + ngram)
# feature_sets.append(feature_word_ngram)
# feature_word_shape = list2dict(default + word + wordshape)
# feature_sets.append(feature_word_shape)
# feature_word_class = list2dict(default + word + classf)
# feature_sets.append(feature_word_class)
# feature_word_disjunctive = list2dict(default + word + disjunctive)
# feature_sets.append(feature_word_disjunctive)

DM = DataManager()
DM.change_pwd()


def run(feature_set, DM=DM):
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=False)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   train_file=DM.train_file,
                   test_file=DM.test_file, result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    crf_test.train()
    crf_test.verify()

if __name__=='__main__':
    # #use feature in the demo list one by one
    # for feature_set in feature_sets:
    #     run(feature_set)

    #use demo features
    feature_demo = features
    run(feature_demo)

    # #use the 3 better feature in demo list
    # feature_word_shape_disjunctive = list2dict(default + word + wordshape + disjunctive)
    # run(feature_word_shape_disjunctive)

    # #use all of demo features except the worst one - class feature
    # feature_exp_class = list2dict(default + word + ngram + wordshape + disjunctive)
    # run(feature_exp_class)

    # #use demo features and two extra new features
    # feature_demo_lemmas_position = features
    # run(feature_demo_lemmas_position)

    '''
        more options of demo feature
            1.useWordPairs=true, useTags=true
            2.lowercaseNGrams=true, useNeighborNGrams=true, conjionShapeNGrams=true
            3.useTypeSeqs3=true, useDisjShape=true
            4.useNextSequences=true, useTaggySequences=true
            5.useDisjunctiveShapeInteraction=true, useWideDisjunctive=true
            6.maxNGramLeng=(4,8)
            7.maxLeft=(1,2,3), maxRight=(1,2,3)
            8.wordShape=chris2useLC
            9.disjunctionWidth=(2,6)
            10.wideDisjunctionWidth=(2,6)
    '''

    # # option 1
    # features_opt1 = copy.deepcopy(features)
    # features_opt1['useWordPairs'] = 'true'
    # features_opt1['useTags'] = 'true'
    # run(features_opt1)
    # del features_opt1
    #
    # # option 2
    # features_opt2 = copy.deepcopy(features)
    # features_opt2['lowercaseNGrams'] = 'true'
    # features_opt2['useNeighborNGrams'] = 'true'
    # features_opt2['conjionShapeNGrams'] = 'true'
    # run(features_opt2)
    # del features_opt2
    #
    # # option 3
    # features_opt3 = copy.deepcopy(features)
    # features_opt3['useTypeSeqs3'] = 'true'
    # features_opt3['useDisjShape'] = 'true'
    # run(features_opt3)
    # del features_opt3
    #
    # # option 4
    # features_opt4 = copy.deepcopy(features)
    # features_opt4['useNextSequences'] = 'true'
    # features_opt4['useTaggySequences'] = 'true'
    # run(features_opt4)
    # del features_opt4
    #
    # # option 5
    # features_opt5 = copy.deepcopy(features)
    # features_opt5['useDisjunctiveShapeInteraction'] = 'true'
    # features_opt5['useWideDisjunctive'] = 'true'
    # run(features_opt5)
    # del features_opt5

    # # option 6
    # features_opt6 = copy.deepcopy(features)
    # for length in range(4, 8 + 1):
    #     features_opt6['maxNGramLeng'] = str(length)
    #     run(features_opt6)
    # del features_opt6

    # # option 7
    # features_opt7 = copy.deepcopy(features)
    # for index in [1, 2]:
    #     features_opt7['maxLeft'] = str(index)
    #     run(features_opt7)
    # del features_opt7

    # # option 8
    # features_opt8 = copy.deepcopy(features)
    # features_opt8['wordShape'] = 'chris4useLC'
    # run(features_opt8)
    # del features_opt8

    # crf_canvas = CanvasFactory.CRFCanvasFactory().produce(result_file=DM.result_file)
    # crf_canvas.load_data()
    # crf_canvas.hist()
