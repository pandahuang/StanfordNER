class ScriptToolkit(object):
    def __init__(self, DM):
        self.DM = DM

    @classmethod
    def get_demo_features(cls):
        features = {
            'useClassFeature': 'true',
            'useWord': 'true',
            'useNGrams': 'true',
            'noMidNGrams': 'true',
            'useDisjunctive': 'true',
            'maxNGramLeng': '6',
            'usePrev': 'true',
            'useNext': 'true',
            'useSequences': 'true',
            'usePrevSequences': 'true',
            'maxLeft': '1',
            'useTypeSeqs': 'true',
            'useTypeSeqs2': 'true',
            'useTypeySequences': 'true',
            'wordShape': 'chris2useLC',
        }
        return features

    def list2dict(self, features, feature):
        return {key: features.get(key) for key in feature if features.has_key(key)}

    def ResultsAndWrongAnswerRecord(self, sout, serr, detail_result):
        results = sout.strip().split('\r')
        isWorng = False
        sents = []
        with open(self.DM.log_wrong_sentences, 'a') as fopen:
            fopen.write(detail_result + '\n')
            fopen.write('----------------------------------------------------------------\n')
        for result in results:
            if result.strip():
                sents.append(result.strip())
                token, label, res = result.split('\t')[0], result.split('\t')[1], result.split('\t')[2]
                if label != res:
                    isWorng = True
            else:
                with open(self.DM.log_wrong_sentences, 'a') as fopen:
                    if sents and isWorng:
                        for sent in sents:
                            fopen.write(sent + '\n')
                        fopen.write('\n')
                sents = []
                isWorng = False
        with open(self.DM.log_wrong_sentences, 'a') as fopen:
            fopen.write('===============================================================================' + '\n')

    def ReadBestAndWorstDataset(self):
        with open(self.DM.train_file) as fopen:
            lines_train = fopen.readlines()
        with open(self.DM.test_file) as fopen:
            lines_test = fopen.readlines()
        return (lines_train, lines_test)

    def WriteBestAndWorstDataset(self, max_accuracy, max_data, min_accuracy, min_data):
        with open(self.DM.log_best_dataset, 'w') as fopen:
            fopen.write(str(max_accuracy) + '\n')
            fopen.write('Train--------------------------------------------------------------------------' + '\n')
            for line in max_data[0]:
                fopen.write(line)
            fopen.write('Test--------------------------------------------------------------------------' + '\n')
            for line in max_data[1]:
                fopen.write(line)
        with open(self.DM.log_worst_dataset, 'w') as fopen:
            fopen.write(str(min_accuracy) + '\n')
            fopen.write('Train--------------------------------------------------------------------------' + '\n')
            for line in min_data[0]:
                fopen.write(line)
            fopen.write('Test--------------------------------------------------------------------------' + '\n')
            for line in min_data[1]:
                fopen.write(line)
