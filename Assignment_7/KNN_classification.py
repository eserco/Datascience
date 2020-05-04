print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# Custom Imports
from utils.preprocessor import preprocess
from utils.scaler import scale
from utils.class_balancer import doUpsamling
from utils.preprocessor import reduceDimentions

print("FIRST EXPERIMENT (1) WITH StandardScaler")
print('\n')
print("========================")
print("Start Get Clean Data/Dim reduction")
# Now we are going to do the dim reduction methods
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels, True)

# Uncomment if you want to do dim reduction with ANOVA
# # Reduce dimensions on the dataset using ANOVA
# sortedAnovaResults, significantValues, reducedDf = reduceDimentions(df,
#     'ANOVA', 0.01, reduce = True)
#
# # Attach the labels of cancer ...again... damn pandas methods
#df = preprocess(reducedDf, labels, False)

print("End Get Clean Data/Dim reduction")
print("========================")
print('\n')

print("========================")
print("Start Up-Sampling")
# Get the columns for future use
rows, columns = df.shape

df_values = df.values

# split to upsample (data = X, labels = y)
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# print dimensions as a bug sprayer
print('The dimension of X is: ', X.shape)
print('The dimension of y is: ', y.shape)

# Upsampling uses random stuff we plug the meaning of life into it
np.random.seed(42)
X, y = doUpsamling(X, y)

# print dimensions as a bug sprayer
print('The dimension of upsampled X is: ', X.shape)
print('The dimension of upsampled y is: ', y.shape)

#y_labels = pd.DataFrame(data=y[:], columns=['cancer'])

print("End Up-Sampling")
print("========================")
print('\n')

print("========================")
print("Start train-test split on the up-sampled dataset")

# Considering removing...
# df_values = df.values
#
# # split to train and test data
# X=df_values[:,0:columns - 1]
# y=df_values[:,[columns - 1]].flatten()

# For bug squasing
print('Dimensions of the vectors to split...')
print('Dimension of X is: ', X.shape)
print('Dimension of y is: ', y.shape)

# Now split the upsampled data for the X-validation
print('Splitting with 70/30...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=42)
print('\n')

# For bug squasing
print('Dimensions of the splited vectors...')
print('Dimension of X_train is: ', X_train.shape)
print('Dimension of y_train is: ', y_train.shape)
print('Dimension of X_test is: ', X_test.shape)
print('Dimension of y_test is: ', y_test.shape)

print("End train-test split on the up-sampled dataset")
print("========================")
print('\n')

print("========================")
print("Start scaling")

# scale training and test data using standar scaler
scaler = preprocessing.StandardScaler().fit(X_train)
# Scale the X_train :)
X_train_scaled = scaler.transform(X_train)
# Scale the test set using the X_train set parameters
X_test_scaled = scaler.transform(X_test)

print('The Dimensions of the train set Scaled are: \n')
print(X_train_scaled.shape)
print('The Dimensions of the test set Scaled are: \n')
print(X_test_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(X_train_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(X_train_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("Start the X-val for hyper-param adjust")
print("========================")

# neighbor list
neighbors = list(range(1,25))

# list with cv test scores
cv_scores = []

for k in neighbors:

	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')

	cv_scores.append(scores.mean())

print("Test score means: \n", cv_scores)

MSE = [1 - x for x in cv_scores]

# determining best k only on test score misclassification
optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.title('MSE X-validation Plot using StandardScaler')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Mean Misclassification Error on the validation sets')
plt.show()

print("End the X-val for hyper-param adjust")
print("========================")
print('\n')

print("========================")
print("Start Model Evaluation")

# Do the kNN with the optimal parameter and check for performance of the test
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn.fit(X_train_scaled, y_train)

# Disregard this is for bug-squasing
print('shape of the x train ', X_train.shape)
print('shape of the y train ', y_train.shape)
print('shape of the x test ', X_test.shape)
print('shape of the y test ', y_test.shape)

# Predict the response using the test data from the split
pred = knn.predict(X_test_scaled)

# evaluate accuracy
print ('The final accuracy score of our model is: ',
	accuracy_score(y_test, pred))

# Confusion matrix of the model
print('The confusion matrix is: ')
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('The True Negatives are: ', tn, ' in percent: ', (tn / (tn + fn)),
	'\n The False positives are: ', fp, ' in percent: ', (fp / (fp + tp)),
	'\n The False Negatives are: ', fn, ' in percent: ', (fn / (fn + tn)),
	'\n The True Positives are: ', tp, ' in percent: ', (tp / (tp + fp)))

# plot accuracy on the test set vs different k's
accuracy_on_test = []
for k in neighbors:
	# Do the kNN with the optimal parameter and check for performance of the test
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_train_scaled, y_train)

	# predict the response
	pred = knn.predict(X_test_scaled)

	acc = accuracy_score(y_test, pred)

	accuracy_on_test.append(acc)

print("Accuracies on the test with different k's is: \n", accuracy_on_test)

# plot misclassification error vs k
plt.plot(neighbors, accuracy_on_test)
plt.title('Accuracy on Test-set Plot using StandardScaler')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy on the test set')
plt.show()

print("End Model Evaluation")
print("========================")
print('\n')

print('\n')
print("===================================================")
print('SECOND EXPERIMENT (2) WITH MIN/MAX SCAILING TECHNIQUE')
print("===================================================")
print('\n')

print('\n')
print("========================")
print("Start Get Clean Data/Dim reduction")
# Now we are going to do the dim reduction methods
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels, True)

# Uncomment if you want to do dim reduction with ANOVA
# # Reduce dimensions on the dataset using ANOVA
# sortedAnovaResults, significantValues, reducedDf = reduceDimentions(df,
#     'ANOVA', 0.01, reduce = True)
#
# # Attach the labels of cancer ...again... damn pandas methods
#df = preprocess(reducedDf, labels, False)

print("End Get Clean Data/Dim reduction")
print("========================")
print('\n')

print("========================")
print("Start Up-Sampling")
# Get the columns for future use
rows, columns = df.shape

df_values = df.values

# split to upsample (data = X, labels = y)
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# print dimensions as a bug sprayer
print('The dimension of X is: ', X.shape)
print('The dimension of y is: ', y.shape)

# Upsampling uses random stuff we plug the meaning of life into it
np.random.seed(42)
X, y = doUpsamling(X, y)

# print dimensions as a bug sprayer
print('The dimension of upsampled X is: ', X.shape)
print('The dimension of upsampled y is: ', y.shape)

#y_labels = pd.DataFrame(data=y[:], columns=['cancer'])

print("End Up-Sampling")
print("========================")
print('\n')

print("========================")
print("Start train-test split on the up-sampled dataset")

# Considering removing...
# df_values = df.values
#
# # split to train and test data
# X=df_values[:,0:columns - 1]
# y=df_values[:,[columns - 1]].flatten()

# For bug squasing
print('Dimensions of the vectors to split...')
print('Dimension of X is: ', X.shape)
print('Dimension of y is: ', y.shape)

# Now split the upsampled data for the X-validation
print('Splitting with 70/30...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=42)
print('\n')

# For bug squasing
print('Dimensions of the splited vectors...')
print('Dimension of X_train is: ', X_train.shape)
print('Dimension of y_train is: ', y_train.shape)
print('Dimension of X_test is: ', X_test.shape)
print('Dimension of y_test is: ', y_test.shape)

print("End train-test split on the up-sampled dataset")
print("========================")
print('\n')

print("========================")
print("Start scaling... using Min Max Scaler")

# scale training and test data using min max scaler
scaler = preprocessing.MinMaxScaler().fit(X_train)

# Scale the X_train :)
X_train_scaled = scaler.transform(X_train)
# Scale the test set using the X_train set parameters
X_test_scaled = scaler.transform(X_test)

print('The Dimensions of the train set Scaled are: \n')
print(X_train_scaled.shape)
print('The Dimensions of the test set Scaled are: \n')
print(X_test_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(X_train_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(X_train_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("Start the X-val for hyper-param adjust")
print("========================")

# neighbor list
neighbors = list(range(1,25))

# list with cv test scores
cv_scores = []

for k in neighbors:

	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')

	cv_scores.append(scores.mean())

print("Test score means: \n", cv_scores)

MSE = [1 - x for x in cv_scores]

# determining best k only on test score misclassification
optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.title('MSE X-validation Plot using MinMaxScaler')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Mean Misclassification Error on the validation sets')
plt.show()

print("End the X-val for hyper-param adjust")
print("========================")
print('\n')

print("========================")
print("Start Model Evaluation")

# Do the kNN with the optimal parameter and check for performance of the test
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn.fit(X_train_scaled, y_train)

# Disregard this is for bug-squasing
print('shape of the x train ', X_train.shape)
print('shape of the y train ', y_train.shape)
print('shape of the x test ', X_test.shape)
print('shape of the y test ', y_test.shape)

# Predict the response using the test data from the split
pred = knn.predict(X_test_scaled)

# evaluate accuracy
print ('The final accuracy score of our model is: ',
	accuracy_score(y_test, pred))

# Confusion matrix of the model
print('The confusion matrix is: ')
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('The True Negatives are: ', tn, ' in percent: ', (tn / (tn + fn)),
	'\n The False positives are: ', fp, ' in percent: ', (fp / (fp + tp)),
	'\n The False Negatives are: ', fn, ' in percent: ', (fn / (fn + tn)),
	'\n The True Positives are: ', tp, ' in percent: ', (tp / (tp + fp)))

# plot accuracy on the test set vs different k's
accuracy_on_test = []
for k in neighbors:
	# Do the kNN with the optimal parameter and check for performance of the test
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_train_scaled, y_train)

	# predict the response
	pred = knn.predict(X_test_scaled)

	acc = accuracy_score(y_test, pred)

	accuracy_on_test.append(acc)

print("Accuracies on the test with different k's is: \n", accuracy_on_test)

# plot misclassification error vs k
plt.plot(neighbors, accuracy_on_test)
plt.title('Accuracy on Test-set Plot using MinMaxScaler')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy on the test set')
plt.show()

print("End Model Evaluation")
print("========================")

print('\n')
print("====================================================")
print('THIRD EXPERIMENT (3) WITH ANOVA AND StandardScaler ')
print("====================================================")
print('\n')

print('\n')
print("========================")
print("Start Get Clean Data/Dim reduction")
# Now we are going to do the dim reduction methods
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels, True)

# Uncomment if you want to do dim reduction with ANOVA
# Reduce dimensions on the dataset using ANOVA
sortedAnovaResults, significantValues, reducedDf = reduceDimentions(df,
    'ANOVA', 0.01, reduce = True)

# Attach the labels of cancer ...again... damn pandas methods
df = preprocess(reducedDf, labels, False)

print("End Get Clean Data/Dim reduction")
print("========================")
print('\n')

print("========================")
print("Start Up-Sampling")
# Get the columns for future use
rows, columns = df.shape

df_values = df.values

# split to upsample (data = X, labels = y)
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# print dimensions as a bug sprayer
print('The dimension of X is: ', X.shape)
print('The dimension of y is: ', y.shape)

# Upsampling uses random stuff we plug the meaning of life into it
np.random.seed(42)
X, y = doUpsamling(X, y)

# print dimensions as a bug sprayer
print('The dimension of upsampled X is: ', X.shape)
print('The dimension of upsampled y is: ', y.shape)

#y_labels = pd.DataFrame(data=y[:], columns=['cancer'])

print("End Up-Sampling")
print("========================")
print('\n')

print("========================")
print("Start train-test split on the up-sampled dataset")

# Considering removing...
# df_values = df.values
#
# # split to train and test data
# X=df_values[:,0:columns - 1]
# y=df_values[:,[columns - 1]].flatten()

# For bug squasing
print('Dimensions of the vectors to split...')
print('Dimension of X is: ', X.shape)
print('Dimension of y is: ', y.shape)

# Now split the upsampled data for the X-validation
print('Splitting with 70/30...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=42)
print('\n')

# For bug squasing
print('Dimensions of the splited vectors...')
print('Dimension of X_train is: ', X_train.shape)
print('Dimension of y_train is: ', y_train.shape)
print('Dimension of X_test is: ', X_test.shape)
print('Dimension of y_test is: ', y_test.shape)

print("End train-test split on the up-sampled dataset")
print("========================")
print('\n')

print("========================")
print("Start scaling... using Standard Scaler")

# scale training and test data using min max scaler
scaler = preprocessing.StandardScaler().fit(X_train)

# Scale the X_train :)
X_train_scaled = scaler.transform(X_train)
# Scale the test set using the X_train set parameters
X_test_scaled = scaler.transform(X_test)

print('The Dimensions of the train set Scaled are: \n')
print(X_train_scaled.shape)
print('The Dimensions of the test set Scaled are: \n')
print(X_test_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(X_train_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(X_train_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("Start the X-val for hyper-param adjust")
print("========================")

# neighbor list
neighbors = list(range(1,25))

# list with cv test scores
cv_scores = []

for k in neighbors:

	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')

	cv_scores.append(scores.mean())

print("Test score means: \n", cv_scores)

MSE = [1 - x for x in cv_scores]

# determining best k only on test score misclassification
optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.title('MSE X-validation Plot using StandardScaler + ANOVA')
plt.ylabel('Mean Misclassification Error on the validation sets')
plt.show()

print("End the X-val for hyper-param adjust")
print("========================")
print('\n')

print("========================")
print("Start Model Evaluation")

# Do the kNN with the optimal parameter and check for performance of the test
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn.fit(X_train_scaled, y_train)

# Disregard this is for bug-squasing
print('shape of the x train ', X_train.shape)
print('shape of the y train ', y_train.shape)
print('shape of the x test ', X_test.shape)
print('shape of the y test ', y_test.shape)

# Predict the response using the test data from the split
pred = knn.predict(X_test_scaled)

# evaluate accuracy
print ('The final accuracy score of our model is: ',
	accuracy_score(y_test, pred))

# Confusion matrix of the model
print('The confusion matrix is: ')
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('The True Negatives are: ', tn, ' in percent: ', (tn / (tn + fn)),
	'\n The False positives are: ', fp, ' in percent: ', (fp / (fp + tp)),
	'\n The False Negatives are: ', fn, ' in percent: ', (fn / (fn + tn)),
	'\n The True Positives are: ', tp, ' in percent: ', (tp / (tp + fp)))

# plot accuracy on the test set vs different k's
accuracy_on_test = []
for k in neighbors:
	# Do the kNN with the optimal parameter and check for performance of the test
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_train_scaled, y_train)

	# predict the response
	pred = knn.predict(X_test_scaled)

	acc = accuracy_score(y_test, pred)

	accuracy_on_test.append(acc)

print("Accuracies on the test with different k's is: \n", accuracy_on_test)

# plot misclassification error vs k
plt.plot(neighbors, accuracy_on_test)
plt.title('Accuracy on Test-set Plot using StandardScaler + ANOVA')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy on the test set')
plt.show()

print("End Model Evaluation")
print("========================")

print('\n')
print("====================================================")
print('FOURTH EXPERIMENT (4) WITH ANOVA AND MinMaxScaler ')
print("====================================================")
print('\n')

print('\n')
print("========================")
print("Start Get Clean Data/Dim reduction")
# Now we are going to do the dim reduction methods
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels, True)

# Uncomment if you want to do dim reduction with ANOVA
# Reduce dimensions on the dataset using ANOVA
sortedAnovaResults, significantValues, reducedDf = reduceDimentions(df,
    'ANOVA', 0.01, reduce = True)

# Attach the labels of cancer ...again... damn pandas methods
df = preprocess(reducedDf, labels, False)

print("End Get Clean Data/Dim reduction")
print("========================")
print('\n')

print("========================")
print("Start Up-Sampling")
# Get the columns for future use
rows, columns = df.shape

df_values = df.values

# split to upsample (data = X, labels = y)
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# print dimensions as a bug sprayer
print('The dimension of X is: ', X.shape)
print('The dimension of y is: ', y.shape)

# Upsampling uses random stuff we plug the meaning of life into it
np.random.seed(42)
X, y = doUpsamling(X, y)

# print dimensions as a bug sprayer
print('The dimension of upsampled X is: ', X.shape)
print('The dimension of upsampled y is: ', y.shape)

#y_labels = pd.DataFrame(data=y[:], columns=['cancer'])

print("End Up-Sampling")
print("========================")
print('\n')

print("========================")
print("Start train-test split on the up-sampled dataset")

# Considering removing...
# df_values = df.values
#
# # split to train and test data
# X=df_values[:,0:columns - 1]
# y=df_values[:,[columns - 1]].flatten()

# For bug squasing
print('Dimensions of the vectors to split...')
print('Dimension of X is: ', X.shape)
print('Dimension of y is: ', y.shape)

# Now split the upsampled data for the X-validation
print('Splitting with 70/30...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=42)
print('\n')

# For bug squasing
print('Dimensions of the splited vectors...')
print('Dimension of X_train is: ', X_train.shape)
print('Dimension of y_train is: ', y_train.shape)
print('Dimension of X_test is: ', X_test.shape)
print('Dimension of y_test is: ', y_test.shape)

print("End train-test split on the up-sampled dataset")
print("========================")
print('\n')

print("========================")
print("Start scaling... using Min Max Scaler")

# scale training and test data using min max scaler
scaler = preprocessing.MinMaxScaler().fit(X_train)

# Scale the X_train :)
X_train_scaled = scaler.transform(X_train)
# Scale the test set using the X_train set parameters
X_test_scaled = scaler.transform(X_test)

print('The Dimensions of the train set Scaled are: \n')
print(X_train_scaled.shape)
print('The Dimensions of the test set Scaled are: \n')
print(X_test_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(X_train_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(X_train_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("Start the X-val for hyper-param adjust")
print("========================")

# neighbor list
neighbors = list(range(1,25))

# list with cv test scores
cv_scores = []

for k in neighbors:

	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')

	cv_scores.append(scores.mean())

print("Test score means: \n", cv_scores)

MSE = [1 - x for x in cv_scores]

# determining best k only on test score misclassification
optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.title('MSE X-validation Plot using MinMaxScaler + ANOVA')
plt.ylabel('Mean Misclassification Error on the validation sets')
plt.show()

print("End the X-val for hyper-param adjust")
print("========================")
print('\n')

print("========================")
print("Start Model Evaluation")

# Do the kNN with the optimal parameter and check for performance of the test
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn.fit(X_train_scaled, y_train)

# Disregard this is for bug-squasing
print('shape of the x train ', X_train.shape)
print('shape of the y train ', y_train.shape)
print('shape of the x test ', X_test.shape)
print('shape of the y test ', y_test.shape)

# Predict the response using the test data from the split
pred = knn.predict(X_test_scaled)

# evaluate accuracy
print ('The final accuracy score of our model is: ',
	accuracy_score(y_test, pred))

# Confusion matrix of the model
print('The confusion matrix is: ')
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('The True Negatives are: ', tn, ' in percent: ', (tn / (tn + fn)),
	'\n The False positives are: ', fp, ' in percent: ', (fp / (fp + tp)),
	'\n The False Negatives are: ', fn, ' in percent: ', (fn / (fn + tn)),
	'\n The True Positives are: ', tp, ' in percent: ', (tp / (tp + fp)))

# plot accuracy on the test set vs different k's
accuracy_on_test = []
for k in neighbors:
	# Do the kNN with the optimal parameter and check for performance of the test
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_train_scaled, y_train)

	# predict the response
	pred = knn.predict(X_test_scaled)

	acc = accuracy_score(y_test, pred)

	accuracy_on_test.append(acc)

print("Accuracies on the test with different k's is: \n", accuracy_on_test)

# plot misclassification error vs k
plt.plot(neighbors, accuracy_on_test)
plt.title('Accuracy on Test-set Plot using MinMaxScaler + ANOVA')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy on the test set')
plt.show()

print("End Model Evaluation")
print("========================")



# # Read the data
# rawData = pd.read_csv("data/data.csv", header=None)
# labels = pd.read_csv("data/labels.csv", header=None)
#
# # Call the custom function
# df = preprocess(rawData, labels, True)
# ###### Check-out our data ########
# #print("The appended resulting dataframe is: \n", df)
#
# # Scale the Dataset
# # Drop the las column Dont need scailing on the labels XD
# df = df.drop(['cancer'], axis=1)
# df = scale(df)
#
# # Get back the cancer label
# df = preprocess(df, labels, False)
#
# # Get the columns for future use
# rows, columns = df.shape
#
# df_values = df.values
# # split to train and test data
# X=df_values[:,0:columns - 1]
# y=df_values[:,[columns - 1]].flatten()
#
# # Split before upsampling
# X_train, X_final_test, y_train, y_final_test = train_test_split(X, y, test_size=0.30,
#  	random_state=45)
#
# # Upsampling uses random stuff we plug the meaning of life into it
# np.random.seed(42)
# X_train, y_train = doUpsamling(X_train, y_train)
#
# y_train_labels = pd.DataFrame(data=y_train[:], columns=['cancer'])
#
# # Convert X_train to a df to attach the labels shape gets the num of columns
# newNames = []
# for i in range(X_train.shape[1]):
# 	newNames.append('x' + str(i+1))
#
# # Transform into DF again
# X_train_dataframe = pd.DataFrame(data=X_train[:], columns=[newNames])
#
# # Call the pre-processor to attach the labels of 'cancer' again
# df = preprocess(X_train_dataframe, y_train_labels, False)
#
# '''
# # Upsampling uses random stuff we plug the meaning of life into it
# np.random.seed(42)
# # Class balancing: upsampling minority class
# df_upsampled_ordered = doUpsamling(df)
# df_upsampled = df_upsampled_ordered.sample(frac=1).reset_index(drop=True)
# df_values = df_upsampled.values
# '''
#
# # scale training data using standar scaler
# #scaler = preprocessing.StandardScaler().fit(X_train)
# #X_train_scaled = scaler.transform(X_train)
#
# # split to train and test data
# X=df_values[:,0:columns - 1]
# y=df_values[:,[columns - 1]].flatten()
#
# # Now split the upsampled data for the X-validation
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
#  	random_state=42)
#
# # neighbor list
# neighbors = list(range(1,25))
#
# # list with cv test scores
# cv_scores = []
#
# print("==========================")
# print("K-NN crossvalidation start")
# for k in neighbors:
#
# 	knn = KNeighborsClassifier(n_neighbors=k)
# 	scores = cross_val_score(knn, X_train, y_train, cv=20, scoring='accuracy')
#
# 	cv_scores.append(scores.mean())
#
# print("K-NN crossvalidation end")
# print("========================")
# print("Test score means: \n", cv_scores)
#
# MSE = [1 - x for x in cv_scores]
#
# # determining best k only on test score misclassification
# optimal_k = neighbors[MSE.index(min(MSE))]
#
# print("The optimal number of neighbors is %d" % optimal_k)
#
# # plot misclassification error vs k
# plt.plot(neighbors, MSE)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Misclassification Error of the validation sets')
# plt.show()
#
# # Do the kNN with the optimal parameter and check for performance of the test
# knn = KNeighborsClassifier(n_neighbors=optimal_k)
#
# # fitting the model
# knn.fit(X_train, y_train)
# # Disregard this is for bug-squasing
# print('shape of the x train ', X_train.shape)
# print('shape of the y train ', y_train.shape)
# print('shape of the x test ', X_test.shape)
# print('shape of the x final test ', X_final_test.shape)
#
# # predict the response
# pred = knn.predict(X_final_test)
#
# # evaluate accuracy
# print ('The final accuracy score of our model is: ',
# 	accuracy_score(y_final_test, pred))
#
# # plot accuracy on the test set vs different k's
# accuracy_on_test = []
# for k in neighbors:
# 	# Do the kNN with the optimal parameter and check for performance of the test
# 	knn = KNeighborsClassifier(n_neighbors=k)
#
# 	# fitting the model
# 	knn.fit(X_train, y_train)
#
# 	# predict the response
# 	pred = knn.predict(X_final_test)
#
# 	acc = accuracy_score(y_final_test, pred)
#
# 	accuracy_on_test.append(acc)
#
# print("Accuracies on the test with different k's is: \n", accuracy_on_test)
#
# # plot misclassification error vs k
# plt.plot(neighbors, accuracy_on_test)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Accuracy on the test set')
# plt.show()
