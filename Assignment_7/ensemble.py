import numpy as np
import pandas as pd
import sklearn.tree as sk
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# from exploratory_analysis import reducedDf
from utils.class_balancer import doUpsamling
from utils.preprocessor import preprocess, reduceDimentions

class Classifier:
    def __init__(self, trainX, trainY, testX, testY):
        pass

    def classify(self, data):
        # Given some training/test data, return a vector of zeros and ones
        # i.e. whether the patient has or hasnt the disease.
        prediction = np.zeros(1)
        return prediction

def majorityvote(data, classifiers):
    # Given data and a set of Classifier's, return the most predicted predictions
    _predictions = [c.classify() for c in classifiers]
    predictions = [pred for pred,_  in _predictions]
    trueYs      = [y    for _   ,y  in _predictions]

    # Assuming the values are in columns
    return mode(np.stack(predictions))[0]

class KNN(Classifier):
    def __init__(self, trainX, trainY, testX, testY, k=1):
        # Gather train and test data for a feature selection phase
        testIdx = len(trainX)
        rawData = pd.DataFrame(data=np.concatenate([trainX, testX]))
        labels = pd.DataFrame(np.concatenate([trainY, testY]))

        df = preprocess(rawData, labels, False)
        # Reduce dimensions on the dataset using ANOVA
        sortedAnovaResults, significantValues, reducedDf = reduceDimentions(df,
            'ANOVA', 0.01, reduce = True)

        # Attach the labels of cancer again
        df = preprocess(reducedDf, labels, False)

        # split to upsample (data = X, labels = y)
        rows, columns = df.shape
        X = df.values[:,0:columns - 1]
        y = df.values[:,[columns - 1]].flatten()

        # Regain train and test sets, scaled and upsampled
        trainX      = X[:testIdx]
        trainY      = y[:testIdx]
        self.testX  = X[testIdx:]
        self.testY  = y[testIdx:]

        # Upsample
        np.random.seed(42)
        trainX, trainY = doUpsamling(trainX, trainY)

        # scale training and test data using min max scaler
        scaler = preprocessing.MinMaxScaler().fit(trainX)
        X_train_scaled  = scaler.transform(trainX)
        self.testX      = scaler.transform(self.testX)

        # Do the kNN with the optimal parameter
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(X_train_scaled, trainY)

    def classify(self):
        # Return predicted labels and true labels
        return self.knn.predict(self.testX), self.testY

class DecisionTree(Classifier):
    def __init__(self, trainX, trainY, testX, testY, **kwargs):
        # Gather train and test data for a feature selection phase
        testIdx = len(trainX)
        rawData = pd.DataFrame(data=np.concatenate([trainX, testX]))
        labels = pd.DataFrame(np.concatenate([trainY, testY]))

        #get first 179 records and label column
        df = preprocess(rawData, labels, False)
        #remove last column before scaling
        df = df.iloc[:, :-1]
        #append class labels again
        df = preprocess(df, labels, False)

        # split to upsample (data = X, labels = y)
        rows, columns = df.shape
        X = df.values[:,0:columns - 1]
        y = df.values[:,[columns - 1]].flatten()

        # Regain train and test sets, scaled and upsampled
        self.trainX      = X[:testIdx]
        self.trainY      = y[:testIdx]
        self.testX  = df.iloc[testIdx:,:-1]
        self.testY  = y[testIdx:]

        # Upsample
        np.random.seed(42)
        self.trainX, self.trainY = doUpsamling(self.trainX, self.trainY)

        #again create labels and convert train sets into dataframes
        newNames = ['x' + str(i+1) for i in range(self.trainX.shape[1])]
        self.trainX = pd.DataFrame(data=self.trainX[:], columns=[newNames])
        self.trainY = pd.DataFrame(data=self.trainY[:], columns=['cancer'])
        #combine training sets before feature selection
        combinedDf = preprocess(self.trainX, self.trainY, False)

        #use ANOVA on training-only set to employ feature reduction
        sortedAnovaResults, significantValues, reducedDf = reduceDimentions(combinedDf,
            'ANOVA', 0.01, reduce = True)

        #append class labels again
        self.reducedDf = reducedDf.join(combinedDf.iloc[:,-1])

        #Build the dtModel with best parameters
        classifier = DecisionTreeClassifier(**kwargs)
        self.dtModel = classifier.fit(self.reducedDf.iloc[:, :-1], self.trainY)

    def classify(self):
        # Return predicted labels and true labels
        lbls = self.reducedDf.iloc[:, :-1].columns
        return self.dtModel.predict(self.testX[lbls]), self.testY

class RandomForest(DecisionTree):
    def __init__(self, trainX, trainY, testX, testY, **kwargs):
        super().__init__(trainX, trainY, testX, testY, **kwargs)

        #Build the dtModel with best parameters
        classifier = RandomForestClassifier(**kwargs)
        self.dtModel = classifier.fit(self.reducedDf.iloc[:, :-1], self.trainY)

# Prepare data
rawData = np.array(pd.read_csv("data/data.csv", header=None))
labels = np.array(pd.read_csv("data/labels.csv", header=None))

#define train and target sets
nLabeledSamples = 179
trainX, valX, trainY, valY = train_test_split(rawData[:nLabeledSamples], labels[:nLabeledSamples], test_size=0.30, random_state = 45)

# Build decisiontree and knn ensemble
decisionTree1 = DecisionTree(trainX, trainY, valX, valY, max_depth=20, min_samples_split= 93, max_leaf_nodes=43, min_samples_leaf=2, random_state=25)
decisionTree2 = DecisionTree(trainX, trainY, valX, valY, max_depth=10, min_samples_split= 50, max_leaf_nodes=30, min_samples_leaf=2, random_state=40)
optim_k = 1 # as determined in KNN_classification
knnClassifier1 = KNN(trainX, trainY, valX, valY, k=optim_k)
knnClassifier2 = KNN(trainX, trainY, valX, valY, k=2)

# Add differently initialised classifiers to the ensemble
classifiers = [decisionTree1, decisionTree2, knnClassifier1, knnClassifier2]
ensemble_knns_trees_pred = majorityvote(valX, classifiers)[0]

# Random Forest
randomForest1 = RandomForest(trainX, trainY, valX, valY, max_depth=20, min_samples_split= 93, max_leaf_nodes=43, min_samples_leaf=2, random_state=25)
randomForest2 = RandomForest(trainX, trainY, valX, valY, max_depth=10, min_samples_split= 50, max_leaf_nodes=30, min_samples_leaf=2, random_state=40)

# Gather results
def accuracy_score(predictedY, trueY):
    err_count = np.sum(predictedY - trueY)
    predicted_count = len(trueY) - err_count
    return predicted_count / len(trueY) * 100

valY = np.squeeze(valY)

# Test the accuracy
ensemble_score = accuracy_score(ensemble_knns_trees_pred, valY)
print("Accuracy for the ensemble classifier on the test data is ", ensemble_score, "%")

prediction = randomForest1.classify()
accuracyScore = accuracy_score(prediction,valY)
print("Accuracy for the Random Forest (optimal) classifier on the test data is ", round(accuracyScore,1), "%")

prediction = randomForest2.classify()
accuracyScore = accuracy_score(prediction,valY)
print("Accuracy for the Random Forest (semi-random) classifier on the test data is ", round(accuracyScore,1), "%")

# Predict unlabeled data amd save
# Now use all data
nLabeledSamples = 179
trainX = rawData[:nLabeledSamples]
trainY = labels[:nLabeledSamples]
testX = rawData[nLabeledSamples:]
testY = np.ones((len(testX),1), dtype=np.int8) #fake labels

# Build decisiontree and knn ensemble
decisionTree1 = DecisionTree(trainX, trainY, testX, testY, max_depth=20, min_samples_split= 93, max_leaf_nodes=43, min_samples_leaf=2, random_state=25)
decisionTree2 = DecisionTree(trainX, trainY, testX, testY, max_depth=10, min_samples_split= 50, max_leaf_nodes=30, min_samples_leaf=2, random_state=40)
optim_k = 1 # as determined in KNN_classification
knnClassifier1 = KNN(trainX, trainY, testX, testY, k=optim_k)
knnClassifier2 = KNN(trainX, trainY, testX, testY, k=2)

# Add differently initialised classifiers to the ensemble
classifiers = [decisionTree1, decisionTree2, knnClassifier1, knnClassifier2]
ensemble_knns_trees_pred = majorityvote(testX, classifiers)[0]
print("Saving KNN/DT ensemble prediction as submission")
np.savetxt("Team_14_clustering.csv", ensemble_knns_trees_pred.astype(int), fmt='%i', delimiter=',')
