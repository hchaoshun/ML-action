from math import log


def get_entropy(dataSet):
    infoLen = len(dataSet)
    labelCount = {}
    for item in dataSet:
        label = item[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1

    result = 0.0
    for label, count in labelCount.items():
        p = float(count) / infoLen
        result += p * log(p, 2)
    return -result


def createDataSet():
    dataSet = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
