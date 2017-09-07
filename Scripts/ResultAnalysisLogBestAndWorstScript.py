import os
from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from ScriptToolkit import ScriptToolkit

features = ScriptToolkit.get_demo_features()

DM = DataManager()
DM.change_pwd()
DM.source_data_file = 'CorpusLabelData_MergedFilter_Update.txt'
DM.remove(DM.log_wrong_sentences)
DM.remove(DM.log_best_dataset)
DM.remove(DM.log_worst_dataset)


def run(feature_set, DM=DM):
    crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                   train_file=DM.train_file, test_file=DM.test_file)
    crf_processor.get_train_data(isRandom=True)
    crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                   source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                   result_file=DM.result_file)
    crf_test.feature_config(features=feature_set)
    sout_train, serr_train, sout_test, serr_test = crf_test.train_and_verify()
    train_datasize, train_time = ScriptToolkit.ParseTrainSoutAndSerr(sout_train, serr_train)
    sent_accuracy, test_datasize, test_time, detail_result = ScriptToolkit.ParseTestSoutAndSerr(sout_test, serr_test)
    ScriptToolkit.LogResult(sent_accuracy, DM.source_data_file, train_datasize, train_time, test_datasize, test_time)
    ScriptToolkit(DM).LogResultsAndWrongAnswer(sout_test, serr_test, detail_result)
    return sent_accuracy


st = ScriptToolkit(DM)

if __name__ == '__main__':
    sent_accuracys = []
    cycle_times = 1
    max_accuracy, min_accuracy = 0.0, 1.0
    max_data, min_data = None, None
    for i in range(cycle_times):
        # use demo features
        feature_demo = features
        sent_accuracy = run(feature_demo)
        sent_accuracys.append(sent_accuracy)
        if sent_accuracy > max_accuracy:
            max_accuracy = sent_accuracy
            max_data = st.ReadBestAndWorstDataset()
        if sent_accuracy < min_accuracy:
            min_accuracy = sent_accuracy
            min_data = st.ReadBestAndWorstDataset()
    st.WriteBestAndWorstDataset(max_accuracy, max_data, min_accuracy, min_data)
    print 'Average sent_accuracy is : %f' % (sum(sent_accuracys) / cycle_times)
