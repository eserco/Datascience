from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import pandas as pd
import numpy as np
import scipy.stats as stats

# Fucntion that performs PCA
def doPCA(scaledDf, dimensionsToReduce):
    # Make a copy not to mess on future stuff
    dfCopy = scaledDf
    # Declare an instance of a PCA object
    pca = PCA(n_components= dimensionsToReduce)
    # Perform the fit of the pca
    pca_components = pca.fit_transform(dfCopy)

    # dataframe out of the resulting pca results
    columnLabels = []
    for i in range(dimensionsToReduce):
        columnLabels.append('PCA ' + str(i+1))

    pca_df = pd.DataFrame(data=pca_components, columns = columnLabels)
    # Get the explained variance
    explainedVariance = pca.explained_variance_ratio_
    # Get the culmutative sum
    # print ('The culmutative sum of the pca components is: \n',
    #     pca.explained_variance_ratio_.cumsum())

    return pca_df, explainedVariance

def doTSNE(scaledDf, p=24, iterations=1000):
    # Because 42 is the answer of the meaning of life
    np.random.seed(42)

    # Now we are going to do t-TSNE
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=p, n_iter=iterations)
    try:
        tsne_results = tsne.fit_transform(scaledDf.values)
    except Exception as e:
        print('You did pass a DF, doing with a np array...')
        tsne_results = tsne.fit_transform(scaledDf)
    print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return tsne_results

# Doing the ANOVA stuff
def doANOVA(scaledDf, variableToTest):
    # Get a dataframe of TRUE of FALSE
    selection = scaledDf['cancer'] == 1
    # Convert to numpy and flatten to get a 1d Vector
    queriedDf = selection.values.flatten()
    # Select just the indices you want with the .loc function
    x1Cancer = scaledDf.loc[selection.values.flatten()]
    x1NotCancer = scaledDf.loc[~selection.values.flatten()]
    # select the column of interest
    x1Cancer = x1Cancer[variableToTest]
    x1NotCancer = x1NotCancer[variableToTest]

    # Run the ANOVA method
    fCurrentVariable, pCurrentVariable = stats.f_oneway(x1Cancer, x1NotCancer)
    return (fCurrentVariable, pCurrentVariable)
