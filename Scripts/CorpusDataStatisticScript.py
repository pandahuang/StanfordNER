from Scripts.ScriptToolkit import ScriptToolkit
from Preprocessing.DataReader import DataReader
from Data.DataManager import DataManager

if __name__ == '__main__':
    # create data manager
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = 'CorpusLabelData_LongAndShortSentences_Sales_v2.txt'

    # create datums
    DR = DataReader(source_data_file=DM.source_data_file)
    DR.standard_read()

    # create toolkits
    ST = ScriptToolkit(DM)

    # analysis
    sentences_amount, tokens_distribution, glabels_distribution = ScriptToolkit.StatisticDatums(DR.Datums)

    # result display
    print 'Sentences Amount is %d' % (sentences_amount)
    print 'Tokens distribution is:'
    print tokens_distribution
    print 'glabels_distribution is:'
    print glabels_distribution
