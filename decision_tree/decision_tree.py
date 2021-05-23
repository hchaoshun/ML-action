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


def majorityCnt(classList):
    classCount = {}
    for val in classList:
        if val not in classList.keys():
            classCount[val] = 0
        classCount[val] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda item : item[1], reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # 第一个终止条件：所有label值都相同, 返回label值
    classList = [sample[-1] for sample in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 第二个终止条件，所有特征都用完，返回label出现次数最多的值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeature = chooseBestFeature(dataSet)
    bestFeatureLab = labels[bestFeature]

    myTree = {bestFeatureLab: {}}
    # todo have to delete it?
    del(labels[bestFeature])

    # 递归构建树
    featureVals = [sample[bestFeature] for sample in dataSet]
    for val in set(featureVals):
        myTree[bestFeatureLab][val] = createTree(splitDataSet(dataSet, bestFeature, val), labels)

    return myTree


if __name__ == '__main__':
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    print(myTree)

