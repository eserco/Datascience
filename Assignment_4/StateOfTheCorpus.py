# NLTK imports
import nltk
from nltk.corpus import state_union
from nltk.probability import DictionaryProbDist

import matplotlib.pyplot as plt
import numpy as np
from util import clean_text

# Import the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('state_union')
nltk.download('stopwords')
nltk.download('punkt')

textnames = state_union.fileids()

# Set up the text to tokenize
bushUnion = state_union.raw('2005-GWBush.txt')

cleanedStemmedWords = clean_text(bushUnion)

# Frequency distribution of the speech
#words = [word for word in state_union.words(text)]
#frequencies = nltk.FreqDist([w for w in words if len(w) > 4 and w.lower() == w])

# word frequencies of the not stemmed document
#frequencies = nltk.FreqDist(stemmedwords)

#for e in frequencies.most_common(10):
#    print (e)
#frequencies.plot(10)

# Perform the frequencies on the cleaned dataset
cleanedFrequencies = nltk.FreqDist(cleanedStemmedWords)
print('The most common stemmed words are: ')
for e in cleanedFrequencies.most_common(10):
    print (e)
cleanedFrequencies.plot(10)

# Using the log transformation
logCleanedFrequencies = DictionaryProbDist(cleanedFrequencies)
logCleanedFrequencies.logprob('secur')

# gather frequencies of words
lenght_of_document= len(cleanedStemmedWords)
arrayFormFrequencies = []
for key in cleanedFrequencies:
    arrayFormFrequencies.append(cleanedFrequencies[key])

# Log of the frequencies
logArrayFormFrequencies = np.log(arrayFormFrequencies)+1

# print(logArrayFormFrequencies)
# Plot of freq, and its log
n, bins, patches = plt.hist(arrayFormFrequencies, 20, density=True, facecolor='green', alpha=0.55)
plt.show()
n, bins, patches = plt.hist(logArrayFormFrequencies, 30, density=True, facecolor='green', alpha=0.55)
plt.show()

# Display all the words
texts = ['2005-GWBush.txt', '2006-GWBush.txt']
for text in texts:
    print("\n\n" + text)
    words = [word for word in state_union.words(text)]

    frequencies = nltk.FreqDist([w for w in words if len(w) > 4 and w.lower() == w])
    for e in frequencies.most_common(10):
        print (e)
    frequencies.plot(10)


# Join all the documents together

# Define where are we going to store the corpus
stateUnion_corpus = []
# Get a list of all the titles of the different documents on the corpus
texts = state_union.fileids()

# Append them into the stateUnion_corpus
for i in texts:
    stateUnion_corpus.append(state_union.raw(i))

# Create an instance of the Vectorizer
vectorizer = TfidfVectorizer(min_df=1, stop_words="english",norm=None)
# Fit the state of the union corpus
X = vectorizer.fit_transform(stateUnion_corpus)

#print (vectorizer.get_feature_names())
# Do the Term Frequency, Document frequency
# @param tf: term frequency
# @param df: document frequency
# @param N: number of documents

np.around(X.toarray(),1)

#XA = X.toarray()
#for row in XA:
#    n = 18
#    top_n_tfidf = (row.argsort()[-n:][::-1])
#    for idx in top_n_tfidf:
#        print (vectorizer.get_feature_names()[idx] + "> " + str(row[idx]))
#    print ()
