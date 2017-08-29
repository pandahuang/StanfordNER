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
                if line.strip().split('\t', 3)[0] == 'test':
                    result = line.strip().split('\t', 3)[1:3]
                    results.append(list2nparray(result))
        except IOError, e:
            print e
        results = np.array(results)
        self.time = results[index:, 0]
        self.accuracy = results[index:, 1]
        pass

    def plot(self):
        plt.figure(figsize=(9, 6))
        X = range(len(self.time))
        ax1 = plt.subplot(211)
        plt.plot(X, self.time, color='lightpink', linewidth=1, label='time')
        plt.legend(loc='upper left', frameon=False)
        ax2 = plt.subplot(212)
        plt.plot(X, self.accuracy, color='lightblue', linewidth=1, label='accuracy')
        plt.legend(loc='upper left', frameon=False)
        plt.show()
        pass

    def hist(self):
        plt.figure(figsize=(9, 6))
        X = range(len(self.time))
        ax1 = plt.subplot(111)
        plt.bar(X, self.accuracy, width=0.35, facecolor='lightskyblue', edgecolor='white')
        plt.title('Accuracy')
        plt.ylim(0.5, 1.0)
        for x, y in zip(X, self.accuracy):
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
