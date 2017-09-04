from Preprocessing import Preprocessor
from Data.DataManager import DataManager
from Preprocessing.TrainAndTestDataPreprocessor import TrainAndTestDataPreprocessor


class Provider(object):
    def produce(self, **kw):
        pass


class CRFProcessorFactory(Provider):
    def produce(self, **kw):
        return Preprocessor.CRFPreprocessor(**kw)


class PreLabelWithPosProcessorFactory(Provider):
    def produce(self, **kw):
        return Preprocessor.PreLabelWithPosPreprocessor(**kw)


class LSTMProcessorFactory(Provider):
    def produce(self, **kw):
        return Preprocessor.LSTMPreprocessor(**kw)


class TrainAndTestDataPreprocessorFactory(Provider):
    def produce(self, **kw):
        return TrainAndTestDataPreprocessor(**kw)


if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = 'CorpusLabelData_MergedFilter_Update.txt'
    crf_factory = CRFProcessorFactory()
    crf_processor = crf_factory.produce(source_data_file=DM.source_data_file, train_file=DM.train_file,
                                        test_file=DM.test_file)
    crf_processor.preprocess()
    crf_processor.get_train_data()

    prepos_factory = PreLabelWithPosProcessorFactory()
    prepos_processor = prepos_factory.produce(train_file=DM.train_file, test_file=DM.test_file)
    prepos_processor.preprocess()
    # prepos_processor.prelabel_with_pos()
    # prepos_processor.prelabel_with_pos_by_sentence()
    prepos_processor.prelabel_with_pos_nocombined_by_sentence()

    # DM.source_data_file = 'CorpusLabelData_MergedFilter_Full.txt'
    # tat_data_factory = TrainAndTestDataPreprocessorFactory()
    # tat_data_processor = tat_data_factory.produce(source_data_file=DM.source_data_file, train_file=DM.train_file,
    #                                               test_file=DM.test_file)
    # tat_data_processor.preprocess()
    # tat_data_processor.reduce_replicate_data()
