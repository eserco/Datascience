import pandas as pd
import numpy as np
import pprint

# User Imports
from utils.dimentionallityReduction import doANOVA

def preprocess(rawData, labels, slice):
    # Change the name of the columns of the training examples
    newNames = []
    for i in range(len(rawData.columns)):
        newNames.append('x' + str(i+1))

    # Change the column names
    rawData.columns=[newNames]
    #print("The basic structure of the data is: \n ", rawData)

    # Change the name of the column to "class"
    labels.columns = ['cancer']
    #print("The first labels are: \n ", labels)

    if(slice == True):
        # Transform the labels into 0 for non diseased and 1 for AML positive
        labels['cancer'] = labels['cancer'].map({
            1: 0,
            2: 1
        })

    # Get values in an arrayform
    valuesToInsert = labels['cancer'].values

    if slice == True:
        # Slice the original data to get just the 179 labeled examples
        slicedRawData = rawData[:179]
        # Add the 'class' column based on the labels
        slicedRawData['cancer'] = valuesToInsert
        # Add the labeled columns to the numpy array
        appendedDf = slicedRawData
    else:
        # Add the 'class' column based on the labels
        rawData['cancer'] = valuesToInsert
        # Add the labeled columns to the numpy array
        appendedDf = rawData

    return appendedDf

def reduceDimentions(scaledDf, method, significance, *positional_parameters, **keyword_parameters):
    # This will be the scaled DF to reduce
    newDf = scaledDf
    # Check if the user wants ANOVA
    if method == 'ANOVA':
        # Get rows and columns
        rows, columns = scaledDf.shape
        # Empty array to save our resultz
        anovaResults = []
        for i in range(columns-1):
            # Save the column index we are working on
            currentIteration = "x" + str(i+1)
            # Call the custom ANOVA function
            currentF, currentP = doANOVA(scaledDf, currentIteration)

            # Save our results on a dictionary format
            currentIndex = i
            curentResults = {
                "currentIndex": currentIndex,
                "fStat": currentF,
                "pValue": currentP
            }
            # Append our results
            anovaResults.append(curentResults)

        # Desc Sort the listed dictionary based on the P-value
        sortedAnova = sorted(anovaResults, key=lambda k: k['pValue'])
        # Pretty print to the console our results
        #pprint.pprint(sortedAnova)

        pValuesArray = []
        indicesArray = []
        #print((sortedAnova[0])['pValue'])
        # Turn the results into an array for a nice Viz
        for i in range(len(sortedAnova)):
            pValuesArray.append(sortedAnova[i].get('pValue'))
            indicesArray.append((sortedAnova[i].get('currentIndex')+1))
        print('The best indices according to ANOVA: ', indicesArray)
        print('Their respective p-values are: ', pValuesArray)

        # Get just the p-values which are < 0.05 our CUTPOINT
        significanceLevel = significance

        significantValues = list(filter(lambda x: x < significanceLevel, pValuesArray))

        print("We have: ", len(significantValues[:]),
            ' significant dimensions using a cutpoint of p-value <', significanceLevel,
            ' according to ANOVA')

        if ('reduce' in keyword_parameters):
            print('===========Reducing dataset ===============')
            newDimensions = len(significantValues)
            print('Reducing to: ', newDimensions , ' dimensions')

            # Change the columns to keep to the df format x1, x2, etc...
            columnsToKeep = []
            for i in range(newDimensions):
                columnsToKeep.append('x' + str(indicesArray[i]))
            print('We are keeping: ', columnsToKeep)
            # Filter the df based on names
            #newDf = newDf[['x1', 'x2']]
            newDf = newDf[columnsToKeep]
            #print('The new transformed dataset via ANOVA is: \n', newDf)

        return sortedAnova, significantValues, newDf
