from Data.DataManager import DataManager
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF
from ScriptToolkit import ScriptToolkit


class ResultAnalysisBasic(object):
    def start(self):
        print 'This is an analyzer, start analyzing...'

    def analyzing(self, DM):
        feature_set = ScriptToolkit.get_demo_features()
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

    def stop(self):
        print 'Analyzing job done...'

    def run(self, DM):
        self.start()
        sent_accuracy = self.analyzing(DM)
        self.stop()
        return sent_accuracy
