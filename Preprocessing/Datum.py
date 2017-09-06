class Datum(object):
    '''
    Do Refactor Here.
    '''

    def __init__(self, doc_id, doc, tokens, glabels, labels):
        self.doc_id = doc_id
        self.doc = doc
        self.tokens = tokens
        self.golden_labels = glabels
        self.labels = labels

    def get_sentence(self):
        return self.doc

    def get_golden_labels(self):
        return self.golden_labels

    def get_labels(self):
        return self.labels
