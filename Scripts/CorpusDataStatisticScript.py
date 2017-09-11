from Scripts.ScriptToolkit import ScriptToolkit
from Preprocessing.DataReader import DataReader
from Data.DataManager import DataManager

DM = DataManager()
DM.change_pwd()
DM.source_data_file = 'CorpusLabelData_LongAndShortSentences_Sales_v2.txt'


def run(DM=DM):
    DR = DataReader(source_data_file=DM.source_data_file)
    DR.standard_read()
    return ScriptToolkit.StatisticDatums(DR.Datums)


if __name__ == '__main__':
    sentences_amount, tokens_distribution, glabels_distribution = run()
    print 'Sentences Amount is %d' % (sentences_amount)
    print 'Tokens distribution is:'
    print tokens_distribution
    print 'glabels_distribution is:'
    print glabels_distribution
