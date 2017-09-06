from Datum import Datum


class DataWriter(object):
    def write_datum(self, datum, file):
        with open(file, 'a') as fopen:
            for token, glabel in zip(datum.tokens, datum.golden_labels):
                fopen.write(token + ' ' + glabel + '\n')
            fopen.write('\n')
