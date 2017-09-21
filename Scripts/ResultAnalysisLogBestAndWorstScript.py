from Data.DataManager import DataManager
from ScriptToolkit import ScriptToolkit
from Preprocessing.DataReader import DataReader
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF

if __name__ == '__main__':
    # create data manager
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = 'CorpusLabelData_MergedFilter.txt'
    DM.remove(DM.log_wrong_sentences)
    DM.remove(DM.log_best_dataset)
    DM.remove(DM.log_worst_dataset)

    # create toolkits
    ST = ScriptToolkit(DM)
    features = ScriptToolkit.get_demo_features()

    # create datums
    DR = DataReader(source_data_file=DM.source_data_file)
    DR.standard_read()

    # analysis
    sent_accuracys = []
    cycle_times = 1
    max_accuracy, min_accuracy = 0.0, 1.0
    max_data, min_data = None, None

    for i in range(cycle_times):
        # data preprocessing
        crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                       train_file=DM.train_file, test_file=DM.test_file)
        crf_processor.get_train_data(DR.Datums)

        # training and testing
        crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                       source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                       result_file=DM.result_file)
        crf_test.feature_config(features=features)
        sout_train, serr_train, sout_test, serr_test = crf_test.train_and_verify()
        train_datasize, train_time = ScriptToolkit.ParseTrainSoutAndSerr(sout_train, serr_train)
        sent_accuracy, test_datasize, test_time, detail_result = ScriptToolkit.ParseTestSoutAndSerr(sout_test, serr_test)
        ScriptToolkit.LogResult(sent_accuracy, DM.source_data_file, train_datasize, train_time, test_datasize, test_time)
        ST.LogResultsAndWrongAnswer(sout_test, serr_test, detail_result)
        sent_accuracys.append(sent_accuracy)

        # get best and worst result
        if sent_accuracy > max_accuracy:
            max_accuracy = sent_accuracy
            max_data = ST.ReadBestAndWorstDataset()
        if sent_accuracy < min_accuracy:
            min_accuracy = sent_accuracy
            min_data = ST.ReadBestAndWorstDataset()
    ST.WriteBestAndWorstDataset(max_accuracy, max_data, min_accuracy, min_data)

    # result display
    print 'Average sent_accuracy is : %f' % (sum(sent_accuracys) / cycle_times)
