class ScriptToolkit(object):
    def get_demo_features(self):
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

    def list2dict(features, feature):
        return {key: features.get(key) for key in feature if features.has_key(key)}

    def ResultsAndWrongAnswerRecord(sout, serr, detail_result):
        results = sout.strip().split('\r')
        isWorng = False
        sents = []
        with open('LogWrongSents.txt', 'a') as fopen:
            fopen.write(detail_result + '\n')
            fopen.write('----------------------------------------------------------------\n')
        for result in results:
            if result.strip():
                sents.append(result.strip())
                token, label, res = result.split('\t')[0], result.split('\t')[1], result.split('\t')[2]
                if label != res:
                    isWorng = True
            else:
                with open('LogWrongSents.txt', 'a') as fopen:
                    if sents and isWorng:
                        for sent in sents:
                            fopen.write(sent + '\n')
                        fopen.write('\n')
                sents = []
                isWorng = False
        with open('LogWrongSents.txt', 'a') as fopen:
            fopen.write('===============================================================================' + '\n')
