from Data.DataManager import DataManager
from ScriptToolkit import ScriptToolkit
from Scripts.ResultAnalysisBasic import ResultAnalysisBasic

if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = 'CorpusLabelData_MergedFilter.txt'
    DM.remove(DM.log_wrong_sentences)
    DM.remove(DM.log_best_dataset)
    DM.remove(DM.log_worst_dataset)
    rab = ResultAnalysisBasic()
    st = ScriptToolkit(DM)
    sent_accuracys = []
    cycle_times = 10
    max_accuracy, min_accuracy = 0.0, 1.0
    max_data, min_data = None, None
    for i in range(cycle_times):
        # use demo features
        sent_accuracy = rab.run(DM)
        sent_accuracys.append(sent_accuracy)
        if sent_accuracy > max_accuracy:
            max_accuracy = sent_accuracy
            max_data = st.ReadBestAndWorstDataset()
        if sent_accuracy < min_accuracy:
            min_accuracy = sent_accuracy
            min_data = st.ReadBestAndWorstDataset()
    st.WriteBestAndWorstDataset(max_accuracy, max_data, min_accuracy, min_data)
    print 'Average sent_accuracy is : %f' % (sum(sent_accuracys) / cycle_times)
