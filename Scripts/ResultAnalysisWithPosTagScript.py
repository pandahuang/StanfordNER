import os
import copy
from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from ScriptToolkit import ScriptToolkit

features = ScriptToolkit.get_demo_features()

DM = DataManager()
DM.change_pwd()
DM.source_data_file = 'CorpusLabelData_MergedFilter_Update.txt'
DM.remove(DM.log_wrong_sentences)


def run(feature_set, DM=DM):
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=True)
    prepos_processor = ProcessorFactory.PreLabelWithPosProcessorFactory().produce(train_file=DM.train_file,
                                                                                  test_file=DM.test_file)
    prepos_processor.prelabel_with_pos_by_sentence(is_combine=True)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                   result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result = crf_test.train_and_verify()
    return sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result


st = ScriptToolkit(DM)

if __name__ == '__main__':
    sent_accuracys = []
    cycle_times = 10
    for i in range(cycle_times):
        # use demo features
        feature_demo = features
        sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result = run(feature_demo)
        st.LogResultsAndWrongAnswer(sout_test, serr_test, detail_result)
        sent_accuracys.append(sent_accuracy)
    print 'Average sent_accuracy is : %f' % (sum(sent_accuracys) / cycle_times)
