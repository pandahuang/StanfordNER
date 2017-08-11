import numpy as np
import matplotlib.pyplot as plt


def list2nparray(data_list):
    return np.array(data_list).astype(np.float)
    pass


class Canvas(object):
    def load_data(self):
        pass

    def plot(self):
        pass

    def hist(self):
        pass

    def scatter(self):
        pass


class CRFCanvas(Canvas):
    def __init__(self, **kw):
        self.result_file = kw.get('result_file')
        self.time = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []

    def load_data(self, index=0):
        results = []
        try:
            fopen = open(self.result_file)
            fopen.readline()
            for line in fopen:
                results.append(list2nparray(line.strip().split('\t')))

        except IOError, e:
            print e
        results = np.array(results)
        self.time = results[index:, 0]
        self.accuracy = results[index:, 1]
        self.precision = results[index:, 2]
        self.recall = results[index:, 3]
        self.f1 = results[index:, 4]
        pass

    def plot(self):
        plt.figure(figsize=(9, 6))
        X = range(len(self.time))
        ax1 = plt.subplot(311)
        plt.plot(X, self.time, color='lightpink', linewidth=1, label='time')
        plt.legend(loc='upper left', frameon=False)
        ax2 = plt.subplot(312)
        plt.plot(X, self.accuracy, color='lightblue', linewidth=1, label='accuracy')
        plt.legend(loc='upper left', frameon=False)
        ax3 = plt.subplot(313)
        plt.plot(X, self.precision, color='lightpink', linewidth=1, label='precision')
        plt.plot(X, self.recall, color='lightblue', linewidth=1, label='recall')
        plt.plot(X, self.f1, color='lightgreen', linewidth=1, label='f1')
        plt.legend(loc='upper left', frameon=False)
        plt.show()
        pass

    def hist(self):
        plt.figure(figsize=(9, 6))
        X = range(len(self.time))
        ax1 = plt.subplot(221)
        plt.bar(X, self.accuracy, width=0.35, facecolor='lightskyblue', edgecolor='white')
        plt.title('Accuracy')
        plt.ylim(0.5, 1.0)
        for x, y in zip(X, self.accuracy):
            plt.text(x + 0.2, y + 0.02, '%.2f' % y, ha='center')
        ax2 = plt.subplot(222)
        plt.bar(X, self.precision, width=0.35, facecolor='lightpink', edgecolor='white')
        plt.title('Precision')
        plt.ylim(0.5, 1.0)
        for x, y in zip(X, self.precision):
            plt.text(x + 0.2, y + 0.02, '%.2f' % y, ha='center')
        ax3 = plt.subplot(223)
        plt.bar(X, self.recall, width=0.35, facecolor='lightgreen', edgecolor='white')
        plt.title('Recall')
        plt.ylim(0.5, 1.0)
        for x, y in zip(X, self.recall):
            plt.text(x + 0.2, y + 0.02, '%.2f' % y, ha='center')
        ax4 = plt.subplot(224)
        plt.bar(X, self.f1, width=0.35, facecolor='yellowgreen', edgecolor='white')
        plt.title('f-value')
        plt.ylim(0.5, 1.0)
        for x, y in zip(X, self.f1):
            plt.text(x + 0.2, y + 0.02, '%.2f' % y, ha='center')
        plt.show()
        pass

    def scatter(self):
        pass


class LSTMCanvas(Canvas):
    def __init__(self, **kw):
        pass

    def load_data(self):
        pass

    def plot(self):
        pass

    def hist(self):
        pass

    def scatter(self):
        pass
