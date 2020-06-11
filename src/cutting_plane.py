from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import numpy as np
import cvxpy as cp
import pdb
from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from scipy.sparse import *
from scipy.sparse.linalg import norm
from sklearn.metrics import f1_score, precision_score, recall_score
import scipy as sp

#stupid pwalk
import sys
sys.path.append('../include/vaidya-walk/code/pyWrapper')
import pwalk

class activeBPM(object):
	""" Active learning classifier: 1 vs. all classifier
	"""

	def __init__(self, opt=["Chebyshev", "Simple", 1000, 1000, 0]):
		[self.center_opt, self.query_opt, self.pool_size, self.max_iter, self.feature_of_interest] = opt  # unpack
		self.C = []  # set of vectors {a_i} used in linear inequalities for cutting plane (-y_i*(a_i^T*x_i)<=0) with y_i = label and x_i = feature
		self.C_labels = []  # labels for linear inequalities used for constraints
		self.prev_ind = []  # store used indices to avoid re-using indices (switch from unlabeled to labeled data)

	def center(self):
		""" Center subalgorithm.
		Args:

		Returns:
			self.w: SVM weights
		"""

		if self.center_opt == "Chebyshev":
			n_constraints = len(self.C)
			A = np.asarray(self.C)  # collect all linear inequalities
			# A = np.hstack((self.C[0].toarray(),self.C[1].toarray()))
			a_norm = np.zeros(n_constraints)
			for i in range(n_constraints):  # calculate 2-norm used in Chebyshev center
				a_norm[i] = norm(A[i], 'fro')
			n_features = np.shape(A[0])[1]
			# solve optimization
			x, R = cp.Variable(n_features), cp.Variable(1)
			constraints = [R >= 0, cp.norm(x) <= 1]
			for i in range(n_constraints):
				constraints.append(-self.C_labels[i]*(A[i]@x) + R*a_norm[i] <= 0)
			prob = cp.Problem(cp.Maximize(R), constraints)
			prob.solve()
			# pdb.set_trace()
			return np.reshape(x.value,(n_features,1))

		elif self.center_opt == "Gravity":
			n_constraints = len(self.C)
			# A = np.hstack((self.C[0].toarray(),self.C[1].toarray()))
			A = self.A
			# A = A[A!=0.0]
			r = 0.5
			sample_inds = np.random.randint(0,A.shape[0],self.pool_size)
			partA = A[sample_inds]
			inds =np.nonzero(partA)
			B = partA[inds]

			inits = np.random.normal(size=partA.shape)

			pdb.set_trace()

			samples=pwalk.generateDikinWalkSamples(inits,partA,np.zeros_like(partA),r, 1).T
			for i in range(1,self.pool_size):
				samples = np.hstack((samples, pwalk.generateDikinWalkSamples(inits,B,np.zeros_like(B),r, (B.shape[0])).T))

			pdb.set_trace()
			g = np.mean(samples,axis=1)
			A[inds] = gg

		elif self.center_opt == "Pressure":
			pass


	def query(self, train_data):
		""" Query subalgorithm. Follows from Tong-Koller p.10.
		Args:
			train_data: input training data

		Returns:
			ind: index of next point to query from unlabeled dataset
		"""

		if self.query_opt == "Simple":
			ind = -1
			# max_val = -1
			min_val = float('inf')
			for i in self.unlabeled_ind:
				if i not in self.prev_ind:
					dist = np.abs(train_data[i,:] @ self.w)
					if dist < min_val:
					# if dist > max_val:
						ind = i
						min_val = dist
						# max_val = dist
		elif self.query_opt == "MaxMin":
			pass
		elif self.query_opt == "Ratio":
			pass

		return ind

	def fit(self, train_data, train_label):
		""" Active learning algorithm fitting.
		Args:
			train_data: input features (7769 documents x 20682 words per document)
			train_label: ground truth labels (7769 documents x 90 labels)

		Returns:
			self.w: SVM weights from active-learning algorithm

		"""
		# arrange labeled/unlabeled indices
		n_examples, n_features = np.shape(train_data)  # (7769, 20682)
		self.unlabeled_ind = np.random.randint(low=0, high=n_examples, size=(self.pool_size))  # contains indices of unlabeled data
		complete_ind = np.arange(0, n_examples)
		self.labeled_ind = [x for x in complete_ind if x not in self.unlabeled_ind]  # contains indices of labeled data

		# initialize algorithm with 2 labeled data (one has +1 label, one has -1 label)
		np.random.shuffle(complete_ind)  # shuffle to randomize
		init_pos_ind, init_neg_ind = -1, -1

		for i in complete_ind:
			tmp_label = train_label[i, self.feature_of_interest]
			if tmp_label == 1 and init_pos_ind == -1:
				init_pos_ind = i
			elif tmp_label == -1 and init_neg_ind == -1:
				init_neg_ind = i
			if init_pos_ind != -1 and init_neg_ind != -1:
				break

		# pdb.set_trace()
		self.A = train_data.toarray()

		# initialize variables
		self.C.append(train_data[init_pos_ind, :])
		self.C.append(train_data[init_neg_ind, :])
		self.C_labels.append(train_label[init_pos_ind, self.feature_of_interest])
		self.C_labels.append(train_label[init_neg_ind, self.feature_of_interest])
		self.prev_ind.append(init_pos_ind)
		self.prev_ind.append(init_neg_ind)
		self.w = np.ones(n_features)/n_features  # not necessary, simply illustrative of size (self.w is constrained within unit norm ball)

		for i in range(self.max_iter):
			print("Iteration: ", i+1)

			# calculate center
			self.w = self.center()
			pdb.set_trace()

			# query next point
			query_ind = self.query(train_data)

			if query_ind == -1:
				break

			# test new cutting plane
			query_label = train_label[query_ind, self.feature_of_interest]  # oracle label
			query_x = train_data[query_ind, :]  # oracle data

			if (query_label*(query_x @ self.w) < 0):  # incorrectly classifies point
				self.C.append(query_x)
				self.C_labels.append(query_label)

			self.prev_ind.append(query_ind)  # change from unlabeled to labeled data point

		return self.w

	def predict(self, test_data):
		""" Apply classification weights.
		Args:
			test_data: input test data

		Returns:
			svm_label: classifier labels {+1, -1}
		"""
		n_examples, n_features = np.shape(test_data)
		svm_label = np.zeros(n_examples)
		for i in range(n_examples):
			svm_label[i] = np.sign(test_data[i,:] @ self.w)

		return svm_label

def main():
	""" Baseline implementation of cutting-plane algorithm.
	Sample 500/1000 pool unlabeled examples from training
	Args:

	Returns:
	"""
	# set seed
	np.random.seed(1)

	# load data
	train_docs_id = pickle.load(open( "./pickled/train_id.p", "rb" ))
	test_docs_id = pickle.load(open( "./pickled/test_id.p", "rb" ))

	traindocs = pickle.load( open("./pickled/vtrain.p","rb"))
	testdocs = pickle.load( open("./pickled/vtest.p","rb"))

	# transform multilabel labels
	mlb = MultiLabelBinarizer()
	train_labels = mlb.fit_transform([reuters.categories(doc_id)
	                                  for doc_id in train_docs_id])  # (7769, 90)
	test_labels = mlb.transform([reuters.categories(doc_id)
	                             for doc_id in test_docs_id])  # (3019, 90)

	# Classifier
	# gather 10 most common labels to develop 1 vs all classifier
	label_sum = np.sum(train_labels, axis=0)
	top_labels = np.unravel_index(label_sum.argsort(axis=None), dims=label_sum.shape)[0][::-1]
	# pdb.set_trace()
	# adjust labels from {+1, 0} to {+1, -1}
	train_labels = 2*train_labels - 1
	test_labels = 2*test_labels - 1


	opt = ["Gravity",		# center option
	# opt = ["Chebyshev",
	 		"Simple",		# margin option
			500, 			#pool size (subset size of train data to use)
			500, 			#max iterations
			top_labels[0]]	#feature of interest to classify
	classifier = activeBPM(opt)

	# pdb.set_trace()

	classifier.fit(traindocs, train_labels)

	# pdb.set_trace()

	predictions = classifier.predict(testdocs)

	# evaluate predictions on specified feature of interest
	error = test_labels[:, top_labels[0]] - predictions
	n_incorrect = np.count_nonzero(error)

	pdb.set_trace()

	# squeeze one shot labels to get metrics
	round_labels = np.argmax(test_labels,axis=1)
	# precision=precision_score(round_labels,predictions,average='micro')
	# recall = recall_score(round_labels,predictions,average='micro')
	# f1 = f1_score(round_labels,predictions,average='micro')
	#
	# print('Micro-average quality numbers')
	# print('Precision: %.4f, Recall %.4f, F1: %.4f '%(precision, recall, f1))
	# # Precision: 0.0010, Recall 0.0010, F1: 0.0010
	#
	# precision=precision_score(round_labels,predictions,average='macro')
	# recall = recall_score(round_labels,predictions,average='macro')
	# f1 = f1_score(round_labels,predictions,average='macro')
	#
	# print('Macro-average quality numbers')
	# print('Precision: %.4f, Recall %.4f, F1: %.4f '%(precision, recall, f1))




if __name__ == "__main__":
	main()
