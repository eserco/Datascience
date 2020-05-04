#INSTRUCTIONS
#Creates decision tree for unlabeled cancer patient data
#download graphviz software to create tree visualization otherwise create_img function doesnt work
#You can either run the three with best parameters or initiate RandomSearchCv to explore how the hyperparameter search takes place
#The hyperparameter search is disabled by default if you want enable it please remove comment symbol # between line 79 and 92.

#import data sets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from random import randint
from sklearn.metrics import accuracy_score,confusion_matrix
from utils.class_balancer import doUpsamling
from utils.preprocessor import preprocess, reduceDimentions
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import graphviz
import io
import pydotplus
import imageio as imgo

# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

#set test to last 180 entries in the rawData set
realTest = rawData.tail(n=180)

#get first 179 records and label column
df = preprocess(rawData, labels, True)

#remove last column before scaling
df = df.iloc[:, :-1]

#append class labels again
prepDf = preprocess(df, labels, False)

#define relevant Features
#relevantFeatures = ['x39','x123','x130','x56','x157','x32'] #best dimensions??

#define train and target sets
train,test =train_test_split(prepDf, test_size=0.30, random_state = 45)
x_train = train.iloc[:, :-1]
x_test = test.iloc[:, :-1]
y_train = train['cancer']
y_test = test['cancer']

#oversample the minority class in x_train and y_train data
np.random.seed(42)
x_train, y_train = doUpsamling(x_train, y_train)

#again create labels and convert train sets into dataframes
newNames = []
for i in range(x_train.shape[1]):
	newNames.append('x' + str(i+1))

x_train = pd.DataFrame(data=x_train[:], columns=[newNames])
y_train = pd.DataFrame(data=y_train[:], columns=['cancer'])

#combine training sets before feature selection
combinedDf = preprocess(x_train, y_train, False)

#use ANOVA on training-only set to employ feature reduction
sortedAnovaResults, significantValues, reducedDf = reduceDimentions(combinedDf,
    'ANOVA', 0.01, reduce = True)

#append class labels again
reducedDf = reducedDf.join(combinedDf.iloc[:,-1])


#Set parameters for hyperparameter search with RandomizedSearchCV
#Below code blcok seperated by "#" sign doesnt need to be run if it takes too much time. The best parameters are already provided below. Check line 96 for details


###########################################################################################
#parameters = {"max_depth":randint(3,50), "min_samples_leaf": randint(1, 100),"max_leaf_nodes": randint(1,50),"min_samples_leaf": randint(2,10), "criterion":['gini'], }
#
#classifier = DecisionTreeClassifier()
#classifier_cv = RandomizedSearchCV (classifier,parameters, cv = 10, n_jobs = -1, n_iter=100000)
#
#dtModel = classifier_cv.fit(reducedDf.iloc[:, :-1],y_train)
#
#print("best parameters based on the hyper-paramatr search:{}".format(classifier_cv.best_params_))
#print("best accuracy score based on 10 fold validation is {}".format(classifier_cv.best_score_))
#
##store param values from dict into list
#parameterValuesList = list(classifier_cv.best_params_.values())
##########################################################################################



#max_depth=20, min_samples_split= 93, max_leaf_nodes=43, min_samples_leaf=2
#The above values are obtained from iterating the RandomizedSearchCV for 100.000 times. The user is free to not use
#provided values and do their own search. In that case code commented right below should be used and code on line 102 should be commented out.
#classifier = DecisionTreeClassifier(max_depth=parameterValuesList[0], min_samples_split= parameterValuesList[1], max_leaf_nodes=parameterValuesList[2], min_samples_leaf=parameterValuesList[3], random_state=25)

#Build the dtModel with best parameters
classifier = DecisionTreeClassifier(max_depth=20, min_samples_split= 93, max_leaf_nodes=43, min_samples_leaf=2, random_state=25)


#fit into model again
dtModel = classifier.fit(reducedDf.iloc[:, :-1],y_train)

#return prediction array of 0 and 1
prediction = classifier.predict(x_test[reducedDf.iloc[:, :-1].columns])

#compute accuracy score
accuracyScore = accuracy_score(y_test,prediction)*100
print("Accuracy for the current decision tree on the test data is ", round(accuracyScore,1), "%")

#create confusion matrix
cmatrix = confusion_matrix(y_test,prediction)

#create column names for realTest dataframe
realTest = rawData.tail(n=180)
newNames = []
for i in range(len(realTest.columns)):
    newNames.append('x' + str(i + 1))

#rename colors of realTest
realTest.columns = newNames

#filter out class column from column names of reducedDf
filteredColumnNames = [i[0] for i in list(reducedDf)]
del filteredColumnNames[-1]

#filter out irrelevant columns from realTest
realTest = realTest[filteredColumnNames]

#predict cancer patients
realPrediction = classifier.predict(realTest)
sum(realPrediction)
print(sum(realPrediction), "cancer patients are found in the unlabeled data")

#create function to output decision tree image
def create_img(decisionTree, path):
    file = io.StringIO()
    export_graphviz(decisionTree, out_file=file,feature_names=filteredColumnNames)
    pydotplus.graph_from_dot_data(file.getvalue()).write_png(path)
    img = imgo.imread(path)
    plt.rcParams['figure.figsize'] = (25,25)
    plt.imshow(img)

create_img(dtModel,'dt_01.png')
