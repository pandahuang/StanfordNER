from Data.DataManager import DataManager
from ScriptToolkit import ScriptToolkit
from Scripts.ResultAnalysisBasic import ResultAnalysisBasic

if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = 'CorpusLabelData_MergedFilter.txt'
    DM.remove(DM.log_wrong_sentences)
    st = ScriptToolkit(DM)
    rab = ResultAnalysisBasic()
    sent_accuracys = []
    cycle_times = 1
    for i in range(cycle_times):
        sent_accuracy = rab.run(DM)
        sent_accuracys.append(sent_accuracy)
    print 'Average sent_accuracy is : %f' % (sum(sent_accuracys) / cycle_times)
