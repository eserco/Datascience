#Importing NLTK and download : tokenizer, tagger, stopwords, corpus
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords, state_union
from nltk.tokenize import word_tokenize
from nltk.collocations import TrigramCollocationFinder
from nltk import pos_tag
nltk.download('state_union')
nltk.download('stopwords')
nltk.download('tagsets')


#make corpusList ready
corpusList = []
for i in range(len(state_union.fileids())):
    corpusList.append(state_union.raw(state_union.fileids()[i]))

#concatanete all raw texts within corpusList
allTexts = " ".join(corpusList)

#get english stop words
stop_words = set(stopwords.words('english'))

#tokenize
tokens = word_tokenize(allTexts)

#tag tokens
tagged = pos_tag(tokens)

#convert tagged tuple into dataframe for the ease of manipulation
tagged = pd.DataFrame(list(tagged), columns=["word","type"])

#turn words into lowercase except NNP and NNPS
for i in range(len(tagged)):
    if tagged["type"][i] in ('NNP','NNPS'):
        pass
    else:
        tagged["word"][i] = tagged["word"][i].lower()

#filter out stopwords and puncuations
stopWordFiltered = [w for w in tagged["word"].values if not w in stop_words]
puncFiltered = [w for w in stopWordFiltered if len(w)>2]

#tag tokens again
filteredTagged = pos_tag(puncFiltered)

#create list for filtering tokens
llist = ['NNP', 'NNPS', 'fiscal', 'year', 'years', '1947', '1946','guests','honored','fellow']

#filter for NNP and NNPS types
filterTypes = [(x,y) for (x,y) in filteredTagged if y not in llist]

#filter for some unwanted words such as fiscal, year, etc
filterExtraWords = [(x,y) for (x,y) in filterTypes if x not in llist]

#find trigrams
finder = TrigramCollocationFinder.from_words(filterExtraWords,3)

#obtain frequency distribution for all trigrams
freqDistTrigram = finder.ngram_fd.items()
freqDistTable = pd.DataFrame(list(freqDistTrigram), columns=['collocations','freq']).sort_values(by='freq', ascending=False)

#returns 10 trigram collocations with highest frequency within corpus
freqDistTable.head(10)
