'''
From the interpreter, run:
improt nltk
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

More info & troubleshooting @ https://www.nltk.org/data.html

This process takes a while--> run it once,
    and your tokenized data will be saved at ./pickled.
'''

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import pickle
import pdb

cachedStopWords = stopwords.words("english")

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token),
                                   words)))

    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and
                               len(token) >= min_length,
                               tokens))
    return filtered_tokens

stop_words = stopwords.words("english")

documents = reuters.fileids()

train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Tokenisation
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             tokenizer=tokenize)

# Learn and transform train documents
vectorized_train_documents = vectorizer.fit_transform(train_docs)
vectorized_test_documents = vectorizer.transform(test_docs)

pickle.dump(vectorized_train_documents, open("./pickled/vtrain.p","wb"))
pickle.dump(vectorized_test_documents, open("./pickled/vtest.p","wb"))
pickle.dump(train_docs_id,open("./pickled/train_id.p","wb"))
pickle.dump(test_docs_id,open("./pickled/test_id.p","wb"))

print('Tokenization and Pickling Successful.')
