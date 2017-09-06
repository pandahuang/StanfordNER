import os
import copy
from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from Visualization import PainterFactory
from ScriptToolkit import ScriptToolkit

features = ScriptToolkit.get_demo_features()

DM = DataManager()
DM.change_pwd()
DM.remove(DM.log_wrong_sentences)


def run(feature_set, DM=DM):
    DM.source_data_file = 'CorpusLabelData_MergedFilter.txt'
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=True)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                   result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result = crf_test.train_and_verify()
    DM.source_data_file = 'CorpusLabelData_MergedFilter_Full.txt'
    tat_data_processor = ProcessorFactory.TrainAndTestDataPreprocessorFactory().produce(source_data_file=DM.source_data_file,
                                                                                        train_file=DM.train_file,
                                                                                        test_file=DM.test_file)
    tat_data_processor.reduce_replicate_data()
    crf_test_f = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                     source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                     result_file=DM.result_file)
    crf_test_f.feature_config(features=feature_set)
    sout_train_f, serr_train_f, sent_accuracy_f, sout_test_f, serr_test_f, detail_result_f = crf_test_f.train_and_verify()
    return sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result, sout_train_f, serr_train_f, sent_accuracy_f, sout_test_f, serr_test_f, detail_result_f


st = ScriptToolkit(DM)

if __name__ == '__main__':
    cycle_times = 10
    sent_accuracys, sent_accuracys_f = [], []
    for i in range(cycle_times):
        # use demo features
        feature_demo = features
        sout_train, serr_train, sent_accuracy, sout_test, serr_test, detail_result, sout_train_f, serr_train_f, sent_accuracy_f, sout_test_f, serr_test_f, detail_result_f = run(
            feature_demo)
        st.LogResultsAndWrongAnswer(sout_test, serr_test, detail_result)
        st.LogResultsAndWrongAnswer(sout_test_f, serr_test_f, detail_result_f)
        sent_accuracys.append(sent_accuracy)
        sent_accuracys_f.append(sent_accuracy_f)
    print 'Average sent_accuracy of former corpus is : %f' % (sum(sent_accuracys) / cycle_times)
    print 'Average sent_accuracy of new corpus is : %f' % (sum(sent_accuracys_f) / cycle_times)
