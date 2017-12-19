from __future__ import division
import operator
from functools import reduce

class NotSeen(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Token '{}' is never seen in the training set.".format(self.value)

class Classifier(object):
    def __init__(self, trainedData):
        super(Classifier, self).__init__()
        self.data = trainedData
        self.defaultProb = 0.1

    def classify(self, tokens):
        documentCount = self.data.getDocCount()
        classes = self.data.getClasses()

        probsOfClasses = {}

        for className in classes:
            tokensProbs = [self.getTokenProb(token, className) for token in tokens]
            try:
                tokenSetProb = reduce(lambda a, b: a * b, (i for i in tokensProbs if i))
            except:
                tokenSetProb = 0
            probsOfClasses[className] = tokenSetProb * self.getPrior(className)
        return sorted(probsOfClasses.items(),
                      key=operator.itemgetter(1),
                      reverse=True)

    def getPrior(self, className):
        return self.data.getClassDocCount(className) / self.data.getDocCount()

    def getTokenProb(self, token, className):
        classDocumentCount = self.data.getClassDocCount(className)

        try:
            tokenFrequency = self.data.getFrequency(token, className)
        except NotSeen as e:
            return None

        if tokenFrequency is None:
            return self.defaultProb

        probablity = tokenFrequency / classDocumentCount
        return probablity


import sys


class TrainedData(object):
    def __init__(self, dict1, dict2):
        self.docCountOfClasses = dict1
        self.frequencies = dict2

    def getClasses(self):
        return self.docCountOfClasses.keys()

    def getDocCount(self):
        return sum(self.docCountOfClasses.values())

    def getClassDocCount(self, className):
        return self.docCountOfClasses.get(className, None)

    def getFrequency(self, token, className):
        if token in self.frequencies:
            # print token
            # print self.frequencies[token]
            tokenFound = self.frequencies[token]
            return tokenFound.get(className) + 5000
        else:
            print "In else"
            return 5000


def train(trainFile, stop_words):
    hamCount = 0
    spamCount = 0
    mapOfWords = {}

    f = open(trainFile, 'r')

    line = f.readline()
    while line:
        line = line.split(" ")

        if (line[1] == "ham"):
            hamCount += 1
        if (line[1] == "spam"):
            spamCount += 1
        for i in range(2, len(line), 2):
            if line[i] in stop_words:
                continue
            if (line[i] not in mapOfWords):
                mapOfWords[line[i]] = {"spam": 0, "ham": 0}
                mapOfWords[line[i]][line[1]] += int(line[i + 1])  # word:{"spam":count}
            else:
                mapOfWords[line[i]][line[1]] += int(line[i + 1])
        line = f.readline()

    mapLabelCounts = {}
    mapLabelCounts["spam"] = spamCount
    mapLabelCounts["ham"] = hamCount

    print mapLabelCounts
    print mapOfWords
    return mapOfWords, mapLabelCounts


def test(testFile, outPutFile, newsClassifier, stop_words):
    f = open(testFile, 'r')
    line = f.readline()
    fout = open(outPutFile, 'w+')
    success = 0
    fail = 0

    count = 0
    while line:
        count += 1
        line = line.split(" ")
        allWords = []
        for i in range(2, len(line), 2):
            if line[i] in stop_words:
                continue
            else:
                allWords.append(line[i])

        classification = newsClassifier.classify(set(allWords))

        if classification[0][1] > classification[1][1] and line[1] == "spam":
            success += 1

        elif classification[0][1] < classification[1][1] and line[1] == "ham":
            success += 1
        else:
            fail += 1

        fout.write(str(classification))
        line = f.readline()

    fout.write("Accuracy" + str(success * 100/count))
    print "Accuracy ", str(success * 100/(count)), "%"


if sys.argv[1] == "-f1" and sys.argv[3] == "-f2" and sys.argv[5] == "-o":
    trainFile = sys.argv[2]
    testFile = sys.argv[4]
    outPutFile = sys.argv[6]
else:
    print "Improper arguments"
    exit(0)


stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
dct1, dct2 = train(trainFile, stop_words)
data = TrainedData(dct2, dct1)
newsClassifier = Classifier(data)
test(testFile, outPutFile, newsClassifier, stop_words)




