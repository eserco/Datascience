import nltk
from nltk.corpus import state_union
from nltk.corpus import gutenberg

import matplotlib.pyplot as plt
import numpy as np
from util import clean_text
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns

sns.set()
sns.set_context('poster')
nltk.download('state_union')
nltk.download('gutenberg')

# Set current corpus (state_union, gutenberg)
corpus = gutenberg

textnames = corpus.fileids()
corpusnames = {gutenberg : 'Gutenberg books', state_union : 'State union documents'}

print("cleaning texts..")
clean_texts = {name : clean_text(corpus.raw(name)) for name in tqdm(textnames)}
print("Calculating frequencies..")
freqs = {name : nltk.FreqDist(text) for name, text in tqdm(clean_texts.items())}

# Create complete vocabulary
wordlist = set()
for freq in freqs.values():
    wordlist.update(freq.keys())
vocabulary = {word : idx for (idx, word) in enumerate(wordlist)}

# Convert texts to wordcount vectors
word_vectors = {name : np.zeros(len(wordlist,)) for name in textnames}
for textname, vector in word_vectors.items():
    for word, freq in freqs[textname].items():
        vector[vocabulary[word]] += freq

# Stack text vectors into single matrix
data = np.stack(word_vectors.values())

# Apply PCA to find linear word combination pc's
_pca_ = PCA(n_components = 2)
_pca_.fit(data)
projection = _pca_.transform(data)

# Get top-5 words in first pc
inverse_vocab = {idx : word for (word, idx) in vocabulary.items() }
sv = _pca_.singular_values_                              # singular values
sv_sorted = np.argsort(sv)[-5:][::-1]            # indices of top 5 sv's
top_words = [inverse_vocab[idx] for idx in sv_sorted]   # top 5 words
word_sv = [(top_words[idx], sv[idx]) for idx in sv_sorted]  # match words w. sv
print("First principal component: \n", word_sv)

# Plot texts on principal component dimension
plt.scatter(projection[:, 0], projection[:, 1], s=12)
plt.title(corpusnames[corpus])
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
