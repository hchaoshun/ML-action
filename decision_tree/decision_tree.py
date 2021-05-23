from math import log


# 获取信息熵
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


# 按照给定特征lie和特征值划分数据集
# 将满足条件的特征剔除
def splitDataSet(dataSet, axis, value):
    result = []
    for data in dataSet:
        if data[axis] == value:
            subSet = data[:axis]
            subSet.extend(data[axis+1:])
            result.append(subSet)
    return result


# 信息增益最大的就是最高的划分特征
# 信息增益是不同熵的差值
def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = get_entropy(dataSet)
    bestInfoGain = 0.0; bestFeature = -1

    for i in range(numFeatures):
        featureList = [sample[i] for sample in dataSet]
        uniqFeature = set(featureList)

        newEntropy = 0.0
        for feature in uniqFeature:
            subDataSet = splitDataSet(dataSet, i, feature)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * get_entropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def createDataSet():
    dataSet = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
