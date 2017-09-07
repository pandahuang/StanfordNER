# -*- coding: utf-8 -*-
import os


def get_abs_path(root, file):
    return os.path.join(root + r'\data', file)


class DataManager(object):
    def __init__(self):
        self.root_path = 'C:\Users\I337906\PycharmProjects\StanfordNER\Data'
        self.source_data_file = 'CorpusLabelData.txt'
        self.train_file = 'train-data.txt'
        self.test_file = 'test-data.txt'
        self.result_file = 'result-record.txt'
        self.path_to_jar = 'stanford-ner.jar'
        self.prop_file = 'cus-crf.prop'
        self.model_file = 'ner-model.ser.gz'
        self.abs_source_data_file = get_abs_path(self.root_path, self.source_data_file)
        self.abs_train_file = get_abs_path(self.root_path, self.train_file)
        self.abs_test_file = get_abs_path(self.root_path, self.test_file)
        self.abs_result_file = get_abs_path(self.root_path, self.result_file)
        self.log_wrong_sentences = 'LogWrongSents.txt'
        self.features_train = 'features-train.txt'
        self.features_test = 'features-test.txt'
        self.log_best_dataset = 'LogBestData.txt'
        self.log_worst_dataset = 'LogWorstData.txt'

    def list_all_files(self):
        for attr, value in DataManager().__dict__.iteritems():
            print '{attr} = {value}'.format(attr=attr, value=value)
        return DataManager().__dict__

    def change_pwd(self, dirname=None):
        if dirname:
            self.root_path = dirname
        pwd = os.getcwd()
        path = self.root_path + r'\data'
        os.chdir(path)
        cwd = os.getcwd()
        print 'Change work directory from {pwd} to {cwd}'.format(pwd=pwd, cwd=cwd)

    pass

    def remove(self, filepath):
        if os.path.exists(os.path.join(os.getcwd(), filepath)):
            os.remove(os.path.join(os.getcwd(), filepath))
        else:
            print 'IOError: %s is not in current work directory.'%filepath


if __name__ == '__main__':
    DM = DataManager()
    data_files = DM.list_all_files()
    DM.change_pwd()
