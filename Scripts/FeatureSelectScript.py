import os
from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from Scripts.ScriptToolkit import ScriptToolkit
from Preprocessing.DataReader import DataReader

if __name__ == '__main__':
    # create data manager
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = 'CorpusLabelData_SalesModule.txt'
    DM.remove(DM.features_train)
    DM.remove(DM.features_test)

    # create datums
    DR = DataReader(source_data_file=DM.source_data_file)
    DR.standard_read()

    # create toolkits
    ST = ScriptToolkit(DM)
    features = ScriptToolkit.get_demo_features()

    # feature setting
    features['printFeatures'] = '1'
    feature_sets = [features]
    # feature_sets = ScriptToolkit.get_custom_features('custom_features.txt')

    train_times = []
    sent_accuracys = []
    for features in feature_sets:
        # data preprocessing
        crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                       train_file=DM.train_file, test_file=DM.test_file)
        crf_processor.get_train_data(DR.Datums, percent=(0, 0), isRandom=False)

        # training and testing
        crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file, train_file=DM.train_file,
                       test_file=DM.test_file, result_file=DM.result_file)
        crf_test.feature_config(features=features)
        sout_train, serr_train = crf_test.train()
        DM.rename('features-1.txt', DM.features_train)
        train_times.append(ScriptToolkit.ParseTrainSoutAndSerr(sout_train, serr_train)[1])
        sout_test, serr_test = crf_test.verify()
        DM.rename('features-1.txt', DM.features_test)
        sent_accuracys.append(ScriptToolkit.ParseTestSoutAndSerr(sout_test, serr_test)[0])

        # result display
        for train_time, sent_accuracy in zip(train_times, sent_accuracys):
            print 'Training Time:%s , Sentence Level Accuracy:%s .' % (train_time, sent_accuracy)
