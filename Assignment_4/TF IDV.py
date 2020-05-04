
# coding: utf-8

# In[2]:


# Check for working notebook
print('Im Working')


# In[1]:


# NLTK imports
import nltk
from nltk.corpus import webtext
from nltk.corpus import state_union
import numpy as np
nltk.download('state_union')
nltk.download('stopwords')
nltk.download('punkt')

print("\n\n")
print('The fields are: ')
print(state_union.fileids())


# # TF.IDV Representation
# Computing the TF.IDV value of each word of each text in the corpus

# In[ ]:


# Compute the TF value of each word from a bag of words (bow)
def computeTF(wordDic, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict


# In[23]:


# Compute the IDF value of each word in a text
def computeIDF(docList):
    import math
    idf = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
                
    for word, val in idfDict.items():
        idfDict[word] = math.log10(((N+1) / (float(val)+1)) + 1)
        
    return idfDict


# In[ ]:


# Compute the TF * IDF value of each element
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

