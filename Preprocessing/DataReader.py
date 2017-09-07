from Datum import Datum


class DataReader(object):
    def __init__(self, **kw):
        self.Datums = []
        self.source_data_file = kw.get('source_data_file')

    def standard_read(self):
        with open(self.source_data_file) as fopen:
            doc_id = 0
            tokens, glabels = [], []
            for line in fopen.readlines():
                line = line.replace(' ', '\t')
                if line.strip():
                    tokens.append(line.strip().split('\t')[0])
                    glabels.append(line.strip().split('\t')[1])
                elif line.strip() == '' and tokens:
                    doc = ' '.join(tokens)
                    datum = Datum(doc_id, doc, tokens, glabels, [])
                    self.Datums.append(datum)
                    doc_id += 1
                    tokens, glabels = [], []

    def add_predict_labels(self, labels):
        pass
