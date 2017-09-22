from Preprocessor import Preprocessor


class ReplaceNullWithOPreprocessor(Preprocessor):
    def __init__(self, **kw):
        self.train_file = kw.get('train_file')
        self.test_file = kw.get('test_file')

    def preprocess(self):
        print 'This is a preprocessor to replace the "NULL" with "O".'

    def replace_null_with_o(self):
        fopen_train = open(self.train_file)
        fopen_test = open(self.test_file)
        lines_train = fopen_train.readlines()
        lines_test = fopen_test.readlines()
        fopen_train.close()
        fopen_test.close()
        with open(self.train_file, 'w') as fopen:
            for line in lines_train:
                if line.strip():
                    if line.strip().split('\t')[1] == 'NULL':
                        line = line.replace('NULL', 'O')
                fopen.write(line)
        with open(self.test_file, 'w') as fopen:
            for line in lines_test:
                if line.strip():
                    if line.strip().split('\t')[1] == 'NULL':
                        line = line.replace('NULL', 'O')
                fopen.write(line)
