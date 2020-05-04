# Porter Stemming
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    # Define tokenization of the words
    words = word_tokenize(text)

    # Stem the words
    stemmer = PorterStemmer()   # Create instance of the porter stemmer
    stemmedwords = [stemmer.stem(word) for word in words]

    # Set-up remove of stop words
    stop_words = set(stopwords.words('english'))

    # Cleanup of the stemmed words
    cleanedStemmedWords = []
    for word in stemmedwords:
        # Not commas periods and applause.
        if word not in  [",",".","``","''",";","?","--",")","(",":","!","applaus"
                        ] and len(word) > 4 and word not in stop_words:
                cleanedStemmedWords.append(word.lower())
    return cleanedStemmedWords
