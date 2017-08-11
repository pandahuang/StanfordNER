from Preprocessing import Preprocessor
from Data.DataManager import DataManager


class Provider(object):
    def produce(self, **kw):
        pass


class CRFProcessorFactory(Provider):
    def produce(self, **kw):
        return Preprocessor.CRFPreprocessor(**kw)


class LSTMProcessorFactory(Provider):
    def produce(self, **kw):
        return Preprocessor.LSTMPreprocessor(**kw)


if __name__ == '__main__':
    DM = DataManager()
    factory = CRFProcessorFactory()
    crf_processor = factory.produce(source_data_file=DM.abs_source_data_file, train_file=DM.abs_train_file,
                                    test_file=DM.abs_test_file)
    crf_processor.preprocess()
    crf_processor.get_train_data()
