from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pdb
import pickle
#finally, my stuff
from estimator import ActiveSVM

'''
Source : https://www.cs.bgu.ac.il/~elhadad/nlp17/ReutersDataset.html
See source for more information/useful stats on dataset.

DATASET INFORMATION:
cats = reuters.categories()
# print("Reuters has %d categories:\n%s" % (len(cats), cats))
# print(reuters.readme())


Data pre-processing based on
https://miguelmalvarez.com/2016/11/07/classifying-reuters-21578-collection-with-python/.
'''
#unpickle shit & save yourself trouble
train_docs_id = pickle.load(open( "./pickled/train_id.p", "rb" ))
test_docs_id = pickle.load(open( "./pickled/test_id.p", "rb" ))

traindocs = pickle.load( open("./pickled/vtrain.p","rb"))
testdocs = pickle.load( open("./pickled/vtest.p","rb"))

# Transform multilabel labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])

pdb.set_trace()

# Classifier

classifier = OneVsRestClassifier(ActiveSVM(random_state=42))
pdb.set_trace()

classifier.fit(traindocs, train_labels)

predictions = classifier.predict(testdocs)
pdb.set_trace()
