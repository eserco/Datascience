print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn import preprocessing
import pprint

# Custom Imports
from utils.preprocessor import preprocess
from utils.preprocessor import reduceDimentions
from utils.scaler import scale
from utils.dimentionallityReduction import doPCA
from utils.dimentionallityReduction import doTSNE
from utils.dimentionallityReduction import doANOVA

print("FIRST EXPERIMENT (1) WITH StandardScaler")
print('\n')
print("========================")
print("Start Get Clean Data")
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function, True if you need to slice the dataset
df = preprocess(rawData, labels, True)
###### Check-out our data ########
#print("The appended resulting dataframe is: \n", df)

print("========================")
print("End Get Clean Data")

print("========================")
print("Start scaling")

# Get the values to array form for the scaler to work
df = df.values
# scale training and test data using standar scaler
scaler = preprocessing.StandardScaler().fit(df)
# Scale the df :)
df_scaled = scaler.transform(df)

print('The Dimensions of the train set Scaled are: \n')
print(df_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(df_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(df_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("========================")
print("Begin t-SNE")
print('\n')

# Do TSNE
tsne_results = doTSNE(df_scaled)

# Pretty Pictures Incoming...
# Save the resulting columns of t-SNE for a plot
x_axis = tsne_results[:,0]
y_axis = tsne_results[:,1]

colors =  df[:,186]

# Plot the results of the t-SNE
fig = plt.figure()
plt.scatter(x_axis, y_axis, c=colors, marker='.')
plt.title("2 dimensions t-SNE representation of the dataset (Standar Scaler)", fontsize=15)
plt.xlabel('x-tsne', fontsize=13)
plt.ylabel('y-tsne', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()

print('\n')
print("End t-SNE")
print("========================")
print('\n')

print('========================')
print("SECOND EXPERIMENT (2) WITH MinMaxScaler")
print('\n')
print("========================")
print("Start Get Clean Data")
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function, True if you need to slice the dataset
df = preprocess(rawData, labels, True)
###### Check-out our data ########
#print("The appended resulting dataframe is: \n", df)

print("========================")
print("End Get Clean Data")

print("========================")
print("Start scaling")

# Get the values to array form for the scaler to work
df = df.values
# scale training and test data using standar scaler
scaler = preprocessing.MinMaxScaler().fit(df)
# Scale the df :)
df_scaled = scaler.transform(df)

print('The Dimensions of the train set Scaled are: \n')
print(df_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(df_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(df_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("========================")
print("Begin t-SNE")
print('\n')

# Do TSNE
tsne_results = doTSNE(df_scaled)

# Pretty Pictures Incoming...
# Save the resulting columns of t-SNE for a plot
x_axis = tsne_results[:,0]
y_axis = tsne_results[:,1]

colors =  df[:,186]

# Plot the results of the t-SNE
fig = plt.figure()
plt.scatter(x_axis, y_axis, c=colors, marker='.')
plt.title("2 dimensions t-SNE representation of the dataset (MinMaxScaler)", fontsize=15)
plt.xlabel('x-tsne', fontsize=13)
plt.ylabel('y-tsne', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()

print('\n')
print("End t-SNE")
print("========================")
print('\n')

print('========================')
print("THIRD EXPERIMENT (3) WITH Anova Feature Selection")
print('\n')
print("========================")
print("Start Get Clean Data and Dim Reductin (ANOVA)")
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function, True if you need to slice the dataset
df = preprocess(rawData, labels, True)

# Reduce dimensions on the dataset using ANOVA
sortedAnovaResults, significantValues, reducedDf = reduceDimentions(df,
    'ANOVA', 0.01, reduce = True)

# Attach the labels of cancer ...again... damn pandas methods
df = preprocess(reducedDf, labels, False)

print("End Get Clean Data Dim Reductin (ANOVA)")
print("========================")
print('\n')

print("========================")
print("Start scaling")

# Get the values to array form for the scaler to work
df = df.values
# scale training and test data using standar scaler
scaler = preprocessing.MinMaxScaler().fit(df)
# Scale the df :)
df_scaled = scaler.transform(df)

print('The Dimensions of the train set Scaled are: \n')
print(df_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(df_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(df_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("========================")
print("Begin t-SNE")
print('\n')

# Do TSNE
tsne_results = doTSNE(df_scaled)

# Pretty Pictures Incoming...
# Save the resulting columns of t-SNE for a plot
x_axis = tsne_results[:,0]
y_axis = tsne_results[:,1]

colors =  df[:,94]

# Plot the results of the t-SNE
fig = plt.figure()
plt.scatter(x_axis, y_axis, c=colors, marker='.')
plt.title("2 dimensions t-SNE representation of the dataset (ANOVA)", fontsize=15)
plt.xlabel('x-tsne', fontsize=13)
plt.ylabel('y-tsne', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()

print('\n')
print("End t-SNE")
print("========================")
print('\n')

print('========================')
print("FOURTH EXPERIMENT (4) WITH PCA top eigenvectors")
print('\n')
print("========================")
print("Start Get Clean Data")
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function, True if you need to slice the dataset
df = preprocess(rawData, labels, True)
###### Check-out our data ########
#print("The appended resulting dataframe is: \n", df)

print("========================")
print("End Get Clean Data")

print("========================")
print("Start scaling")

# Get the values to array form for the scaler to work
df = df.values
# scale training and test data using standar scaler
df = scale(df)

print('The Dimensions of the train set Scaled are: \n')
print(df_scaled.shape)
print('The means of the dimensions after the scailing are: \n')
meansArray = np.mean(df_scaled, axis=0)
print(meansArray, '\n')
print('The stds of the dimensions after the scailing are: \n')
stdsArray = np.std(df_scaled, axis=0)
print(stdsArray)

print("End scaling")
print("========================")
print('\n')

print("========================")
print("Begin PCA reduction")
print('\n')

# Perform PCA with 2 var on the scaledDf
# Count the number of columns
rows, columns = df.shape
print('The number of columns is: ',columns)

# Remove the last column which contains the label results from the scaled df
df = df.drop(['x'+ str(columns)], axis=1)

pca_df, explainedVar = doPCA(df, 20)
print('The pca is: \n', pca_df,
    '\n the culmutative sum of the variance explained is: ',
    explainedVar.cumsum())

print("End PCA reduction")
print("========================")
print('\n')

print("========================")
print("Begin t-SNE")
print('\n')

# Do TSNE
tsne_results = doTSNE(df_scaled)

# Pretty Pictures Incoming...
# Save the resulting columns of t-SNE for a plot
x_axis = tsne_results[:,0]
y_axis = tsne_results[:,1]

# Get colors the lazy way TODO fix this XD im tired maaan
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function, True if you need to slice the dataset
df_colors = preprocess(rawData, labels, True)
df_colors = df_colors.values
colors =  df_colors[:,186]
# End lazyness....

# Plot the results of the t-SNE
fig = plt.figure()
plt.scatter(x_axis, y_axis, c=colors, marker='.')
plt.title("2 dimensions t-SNE representation of the dataset (PCA)", fontsize=15)
plt.xlabel('x-tsne', fontsize=13)
plt.ylabel('y-tsne', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()

print('\n')
print("End t-SNE")
print("========================")
print('\n')
