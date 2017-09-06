from Preprocessor import Preprocessor
import nltk


def open_file(path):
    fopen = open(path)
    lines = fopen.readlines()
    fopen.close()
    return lines


class PreLabelWithPosPreprocessor(Preprocessor):
    def __init__(self, **kw):
        self.train_file = kw.get('train_file')
        self.test_file = kw.get('test_file')

    def preprocess(self):
        print 'This is PreLabel preprocessor using pos tag.'

    def combine_pos_tag(self, pos_tag):
        noun = ['NN', 'NNS', 'NNP', 'NNPS']
        adjective = ['JJ', 'JJR', 'JJS']
        adverb = ['RB', 'RBR', 'RBS']
        verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        wh = ['WDT', 'WP', 'WRB']
        if pos_tag in noun:
            return 'NN'
        elif pos_tag in adjective:
            return 'JJ'
        elif pos_tag in adverb:
            return 'RB'
        elif pos_tag in verb:
            return 'VB'
        elif pos_tag in wh:
            return 'WP'
        else:
            return pos_tag

    def prelabel_with_pos_all(self, datums, is_combined=False, file=''):
        pass

    def prelabel_with_pos_all(self, is_combined=False, file=''):
        files = []
        if file == '':
            files.append(self.train_file)
            files.append(self.test_file)
        else:
            files.append(file)
        for file in files:
            lines = open_file(file)
            with open(file, 'w') as fopen:
                for line in lines:
                    if line.strip():
                        token = line.strip().split(' ')[0]
                        label = line.strip().split(' ')[1]
                        nlabel = nltk.pos_tag([token])
                        if is_combined:
                            fopen.write(token + ' ' + self.combine_pos_tag(nlabel[0][1]) + '\n')
                        else:
                            fopen.write(token + ' ' + nlabel[0][1] + '\n')
                    else:
                        fopen.write(line)

    def prelabel_with_pos_by_sentence(self, datums, is_combine=False, file='', former_labels=['null']):
        pass

    def prelabel_with_pos_by_sentence(self, is_combine=False, file='', former_labels=['null']):
        files = []
        if file == '':
            files.append(self.train_file)
            files.append(self.test_file)
        else:
            files.append(file)
        for file in files:
            lines = open_file(file)
            with open(file, 'w') as fopen:
                tokens, labels = [], []
                for line in lines:
                    if line.strip():
                        token = line.strip().split(' ')[0]
                        label = line.strip().split(' ')[1]
                        tokens.append(token)
                        labels.append(label)
                    else:
                        pos_tags = nltk.pos_tag(tokens)
                        for (token, pos_tag), label in zip(pos_tags, labels):
                            if label.lower() in former_labels:
                                if is_combine:
                                    fopen.write(token + ' ' + self.combine_pos_tag(pos_tag) + '\n')
                                else:
                                    fopen.write(token + ' ' + pos_tag + '\n')
                            else:
                                fopen.write(token + ' ' + label + '\n')
                        fopen.write(line)
                        tokens, labels = [], []
