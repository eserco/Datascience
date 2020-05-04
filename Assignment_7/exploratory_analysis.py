print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pprint
import seaborn as sns

# Custom Imports
from utils.preprocessor import preprocess
from utils.preprocessor import reduceDimentions
from utils.scaler import scale
from utils.dimentionallityReduction import doPCA
from utils.dimentionallityReduction import doTSNE
from utils.dimentionallityReduction import doANOVA

# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function, True if you need to slice the dataset
df = preprocess(rawData, labels, True)
###### Check-out our data ########
print("The appended resulting dataframe is: \n", df)

# Perform scailing on the data
scaledDf = scale(df)
print(scaledDf)

# Perform PCA with 2 var on the scaledDf
# Count the number of columns
rows, columns = scaledDf.shape
print('The number of columns is: ',columns)

# Remove the last column which contains the label results from the scaled df
scaledDf = scaledDf.drop(['x'+ str(columns)], axis=1)
print("The scaled resulting dataframe is: \n", scaledDf)

pca_df, explainedVar = doPCA(scaledDf, 20)
print('The pca is: \n', pca_df,
    '\n the culmutative sum of the variance explained is: ',
    explainedVar.cumsum())

# Plot our result for mad-viewz
# Plot config to get the color labels for the scatter plots
plot = df.values
colors =  plot[:,186]

fig = plt.figure()
plt.title("2 dimensions PCA representation of the dataset", fontsize=15)
plt.xlabel('PCA_1', fontsize=13)
plt.ylabel('PCA_2', fontsize=13)
plt.scatter(pca_df['PCA 1'],pca_df['PCA 2'], c=colors, marker='.')
plt.legend()
fig.set_size_inches(10, 10, forward=True)
plt.show()

# Set label names for the plot
x_cords = []
for i in range(len(explainedVar[:])):
    x_cords.append(i + 1)

# plot the PCA scree Plot
fig, ax = plt.subplots()
ax.plot(x_cords, explainedVar, 'o-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# plot the PCA culmutative variance explained Plot
fig, ax = plt.subplots()
ax.plot(x_cords, explainedVar.cumsum(), 'o-')
plt.title('Culmutative Variance Plot')
plt.xlabel('Principal Component')
plt.ylabel("`%` of the Var explained")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# Do TSNE (try p=2, 30)
tsne_results = doTSNE(scaledDf, 30, 5000)
# Save the resulting columns of t-SNE for a plot
x_axis = tsne_results[:,0]
y_axis = tsne_results[:,1]

# Plot the results of the t-SNE
plt.scatter(x_axis, y_axis, c=colors, marker='.')
plt.title("2 dimensions t-SNE representation of the dataset", fontsize=15)
plt.xlabel('x-tsne', fontsize=13)
plt.ylabel('y-tsne', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()

# Moar dimentionallity Reduction via ANOVA
# Attach labels to the scaled dataset again...
scaledDf = preprocess(scaledDf, labels, False)

# Call the Anova function
sortedAnovaResults, significantValues, reducedDf = reduceDimentions(scaledDf, 'ANOVA',
    0.01, reduce= True)

# Plot for fun and profit
fig, ax = plt.subplots()
x_cords = np.array(range(len(significantValues[:]))) + 1
ax.plot(x_cords, significantValues, 'o-')
plt.title('P-Values plot')
plt.xlabel('Dimension')
plt.ylabel("p-value")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

print("The reduced df according to anova p-values is: \n", reducedDf)

# Move the seaborn colors to not interfere with plotlib ones
sns.set()
# Boxplot overview of 3 best, 3 worst significant, and three insignificant features
signif_idxs = set(reducedDf.keys().get_level_values(0))
insignif_idxs = set(scaledDf.keys().get_level_values(0)).difference(signif_idxs)
np.random.shuffle(list(insignif_idxs)) #pick random insignificant feats

# Get feature value arrays for each feature
good_select = [scaledDf['x' + str(res['currentIndex'])] for res in sortedAnovaResults[:3]]
bad_select = [scaledDf['x' + str(res['currentIndex'])] for res in sortedAnovaResults[-3:]]
not_selected = [scaledDf[lbl] for lbl in list(insignif_idxs)[:3] if lbl != 'cancer']

# gather data and labels for boxplot
selection = [np.squeeze(arr) for arr in good_select + bad_select + not_selected]
lbls = ['best\n1','best\n2','best\n3','signif\n4','signif\n5', 'signif\n6','bad\n7','bad\n8','bad\n9']
df2 = pd.DataFrame(data=np.array(selection).transpose(), columns=lbls)
plot_lbls = [lbls[i][:-1] + str(res['currentIndex']) for i, res in enumerate(sortedAnovaResults[:3])]
plot_lbls += [lbls[i+3][:-1] + str(res['currentIndex']) for i, res in enumerate(sortedAnovaResults[-3:])]
plot_lbls += [lbls[i+6][:-1] + lbl[1:] for i,lbl in enumerate(list(insignif_idxs)[:3]) if lbl != 'cancer']

# boxplot
sns.set_context('poster')
plt.figure()
hues = [color for color in sns.color_palette("colorblind", 3) for _ in range(3)]
ax = sns.boxplot(data=df2, palette=hues)
plt.xticks(range(9), plot_lbls)
ax.set(ylabel='scaled values')
plt.title("feature value distributions")
plt.show()
