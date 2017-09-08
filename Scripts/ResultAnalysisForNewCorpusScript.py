from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from ScriptToolkit import ScriptToolkit
from Preprocessing.DataReader import DataReader

features = ScriptToolkit.get_demo_features()

DM = DataManager()
DM.change_pwd()
DM.remove(DM.log_wrong_sentences)


def run(features, DM):
    DM.source_data_file = 'CorpusLabelData_MergedFilter.txt'
    DR = DataReader(source_data_file=DM.source_data_file)
    DR.standard_read()
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(DR.Datums)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                   result_file=DM.result_file)
    crf_test.feature_config(features=features)
    sout_train, serr_train, sout_test, serr_test = crf_test.train_and_verify()
    train_datasize, train_time = ScriptToolkit.ParseTrainSoutAndSerr(sout_train, serr_train)
    sent_accuracy, test_datasize, test_time, detail_result = ScriptToolkit.ParseTestSoutAndSerr(sout_test, serr_test)
    ScriptToolkit.LogResult(sent_accuracy, DM.source_data_file, train_datasize, train_time, test_datasize, test_time)
    ScriptToolkit(DM).LogResultsAndWrongAnswer(sout_test, serr_test, detail_result)

    DM.source_data_file = 'CorpusLabelData_MergedFilter_Full.txt'
    tat_data_processor = ProcessorFactory.TrainAndTestDataPreprocessorFactory().produce(source_data_file=DM.source_data_file,
                                                                                        train_file=DM.train_file,
                                                                                        test_file=DM.test_file)
    tat_data_processor.reduce_replicate_data()
    crf_test_f = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                     source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                     result_file=DM.result_file)
    crf_test_f.feature_config(features=features)
    sout_train_f, serr_train_f, sout_test_f, serr_test_f = crf_test_f.train_and_verify()
    train_datasize_f, train_time_f = ScriptToolkit.ParseTrainSoutAndSerr(sout_train_f, serr_train_f)
    sent_accuracy_f, test_datasize_f, test_time_f, detail_result_f = ScriptToolkit.ParseTestSoutAndSerr(sout_test_f, serr_test_f)
    ScriptToolkit.LogResult(sent_accuracy_f, DM.source_data_file, train_datasize_f, train_time_f, test_datasize_f, test_time_f)
    ScriptToolkit(DM).LogResultsAndWrongAnswer(sout_test_f, serr_test_f, detail_result_f)
    return sent_accuracy, sent_accuracy_f


st = ScriptToolkit(DM)

if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    DM.remove(DM.log_wrong_sentences)

    features = ScriptToolkit.get_demo_features()
    cycle_times = 1
    sent_accuracys, sent_accuracys_f = [], []
    for i in range(cycle_times):
        # use demo features
        sent_accuracy, sent_accuracy_f = run(features, DM)
        sent_accuracys.append(sent_accuracy)
        sent_accuracys_f.append(sent_accuracy_f)
    print 'Average sent_accuracy of former corpus is : %f' % (sum(sent_accuracys) / cycle_times)
    print 'Average sent_accuracy of new corpus is : %f' % (sum(sent_accuracys_f) / cycle_times)
