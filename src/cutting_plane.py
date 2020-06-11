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
from sklearn.preprocessing import normalize
import argparse 
import warnings
# import pwalk
# from scipy.spatial import ConvexHull

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

		elif self.center_opt == "Analytic":
			n_constraints = len(self.C)
			A = np.asarray(self.C)  # collect all linear inequalities 
			a_norm = np.zeros(n_constraints)  
			for i in range(n_constraints):  # calculate 2-norm used in Chebyshev center 
				a_norm[i] = norm(A[i], 'fro')
			n_features = np.shape(A[0])[1]
			# solve optimization 
			x = cp.Variable(n_features)
			constraints = [cp.norm(x) <= 1]
			cumsum = 0
			for i in range(n_constraints):
				cumsum += -cp.log(-1*(-self.C_labels[i]*(A[i]@x)))
			prob = cp.Problem(cp.Minimize(cumsum), constraints)
			prob.solve()
			# pdb.set_trace()
			return np.reshape(x.value,(n_features,1))

		elif self.center_opt == "Gravity":
			# direct bayes point machine with tetrahedron approximation
			n_constraints = len(self.C)
			A = np.asarray(self.C)  # collect all linear inequalities 
			n_features = np.shape(A[0])[1]

			# construct tetrahedron points at edges of hyperplane 
			points = np.zeros((n_constraints+1, n_features))
			for i in range(n_constraints):
				if norm(A[i], 'fro') != 0:
					points[i,:] = self.C_labels[i]*A[i].todense()/norm(A[i], 'fro')

			centroid = np.mean(points, axis=0)			
			# pdb.set_trace()
			return centroid

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
			min_val = 1e16
			for i_ind in self.unlabeled_ind:
				if i_ind not in self.prev_ind:
					dist = np.abs(train_data[i_ind,:] @ self.w)
					if dist < min_val:
						ind = i_ind
						min_val = dist

		elif self.query_opt == "MaxMin":
			ind = -1
			margin = -1
			# setup optimization problems
			n_constraints = len(self.C)
			A = np.asarray(self.C)  # collect all linear inequalities 
			a_norm = np.zeros(n_constraints+1)  
			for i in range(n_constraints):  # calculate 2-norm used in Chebyshev center 
				a_norm[i] = norm(A[i], 'fro')
			n_features = np.shape(A[0])[1]
			# solve optimization 
			x, R = cp.Variable(n_features), cp.Variable(1)
			constraints = [R >= 0, cp.norm(x) <= 1]
			for i in range(n_constraints):
				constraints.append(-self.C_labels[i]*(A[i]@x) + R*a_norm[i] <= 0)
			constraints.append(0)  # append placeholder 

			# pick 1/N of the unlabeled dataset for computation efficiency
			rand_unlabeled_ind = self.unlabeled_ind[np.random.randint(low=0, high=len(self.unlabeled_ind), size=round(len(self.unlabeled_ind)/32))]
			for i_ind in rand_unlabeled_ind:
				# print("MaxMin index: ", i_ind)
				if i_ind not in self.prev_ind:					
					A_tmp = train_data[i_ind,:]  # add potential query point 
					a_norm[-1] = norm(A_tmp, 'fro')

					# calculate margin if x is +1 
					constraints[-1] = (1*A_tmp@x + R*a_norm[-1] <= 0)  # if label was +1
					prob = cp.Problem(cp.Maximize(R), constraints)
					prob.solve()
					pos_margin = R.value
					if prob.status == "infeasible":
						continue

					# calculate margin if x is -1
					constraints[-1] = (-1*A_tmp@x + R*a_norm[-1] <= 0)
					prob = cp.Problem(cp.Maximize(R), constraints)
					prob.solve()
					neg_margin = R.value
					if prob.status == "infeasible":
						continue

					tmp_margin = np.amin(np.array([pos_margin, neg_margin]))
					if tmp_margin > margin:
						margin = tmp_margin
						ind = i_ind

		elif self.query_opt == "Ratio":
			ind = -1
			margin = -1
			# setup optimization problems
			n_constraints = len(self.C)
			A = np.asarray(self.C)  # collect all linear inequalities 
			a_norm = np.zeros(n_constraints+1)  
			for i in range(n_constraints):  # calculate 2-norm used in Chebyshev center 
				a_norm[i] = norm(A[i], 'fro')
			n_features = np.shape(A[0])[1]
			# solve optimization 
			x, R = cp.Variable(n_features), cp.Variable(1)
			constraints = [R >= 0, cp.norm(x) <= 1]
			for i in range(n_constraints):
				constraints.append(-self.C_labels[i]*(A[i]@x) + R*a_norm[i] <= 0)
			constraints.append(0)  # append placeholder 

			# pick 1/N of the unlabeled dataset for computation efficiency
			rand_unlabeled_ind = self.unlabeled_ind[np.random.randint(low=0, high=len(self.unlabeled_ind), size=round(len(self.unlabeled_ind)/32))]
			for i_ind in rand_unlabeled_ind:
				# print("Ratio index: ", i_ind)
				if i_ind not in self.prev_ind:					
					A_tmp = train_data[i_ind,:]  # add potential query point 
					a_norm[-1] = norm(A_tmp, 'fro')

					# calculate margin if x is +1 
					constraints[-1] = (1*A_tmp@x + R*a_norm[-1] <= 0)  # if label was +1
					prob = cp.Problem(cp.Maximize(R), constraints)
					prob.solve()
					pos_margin = R.value
					if prob.status == "infeasible":
						continue

					# calculate margin if x is -1
					constraints[-1] = (-1*A_tmp@x + R*a_norm[-1] <= 0)
					prob = cp.Problem(cp.Maximize(R), constraints)
					prob.solve()
					neg_margin = R.value
					if prob.status == "infeasible":
						continue

					tmp_margin = np.amin(np.array([neg_margin/pos_margin, pos_margin/neg_margin]))
					if tmp_margin > margin:
						margin = tmp_margin
						ind = i_ind

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

		# initialize variables
		self.C.append(train_data[init_pos_ind, :])
		self.C.append(train_data[init_neg_ind, :])
		self.C_labels.append(train_label[init_pos_ind, self.feature_of_interest])
		self.C_labels.append(train_label[init_neg_ind, self.feature_of_interest])
		self.prev_ind.append(init_pos_ind)
		self.prev_ind.append(init_neg_ind)
		self.w = np.ones(n_features)/n_features  # not necessary, simply illustrative of size (self.w is constrained within unit norm ball) 

		for i in range(self.max_iter):
			# print("Iteration: ", i+1)

			# calculate center 
			self.w = self.center()

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

def main(args):
	""" Baseline implementation of cutting-plane algorithm. Sample 500/1000 pool unlabeled examples from training
	Args:
	
	Returns:
	"""

	# unpack arguments
	if args.center == 0:
		center_opt = "Chebyshev" 
	elif args.center == 1:
		center_opt = "Gravity"

	if args.query == 0:
		query_opt = "Simple"
	elif args.query == 1:
		query_opt = "MaxMin"
	elif args.query == 2:
		query_opt = "Ratio"

	max_iter = args.max_iter
	pool_size = args.pool_size
	label_number = args.label

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

	# adjust labels from {+1, 0} to {+1, -1}
	train_labels = 2*train_labels - 1
	test_labels = 2*test_labels - 1

	# opt = ["Chebyshev", "Simple", 500, 500, top_labels[1]]  
	# center option, margin option, pool size (subset size of train data to use), max iterations, feature of interest to classify
	opt = [center_opt, query_opt, pool_size, max_iter, top_labels[label_number]]  
	# center option, margin option, pool size (subset size of train data to use), max iterations, feature of interest to classify
	classifier = activeBPM(opt)

	# pdb.set_trace()

	classifier.fit(traindocs, train_labels)

	# pdb.set_trace()

	predictions = classifier.predict(testdocs)

	# evaluate predictions on specified feature of interest
	error = test_labels[:, top_labels[label_number]] - predictions
	n_incorrect = np.count_nonzero(error)
	accuracy = 1 - (n_incorrect / np.shape(test_labels)[0])

	# print("Results: ", "\n", "Accuracy: ", accuracy, "\n", "Queries: ", max_iter, "\n", "Number of cuts: ", len(classifier.C))

	return accuracy, len(classifier.C)  # return accuracy and number of cuts

if __name__ == "__main__":

	# remove deprecation warnings 
	warnings.filterwarnings("ignore", category=DeprecationWarning) 

    # parse settings
	parser = argparse.ArgumentParser(description='Active learning parameters')

	# environment settings
	parser.add_argument('--seed', type=int, default=0, metavar='N', help='seed value')
	parser.add_argument('--center', type=int, default=0, metavar='N', help='center option (0: Chebyshev, 1: Gravity)')
	parser.add_argument('--query', type=int, default=0, metavar='N', help='query option (0: Simple, 1: MaxMin, 2: Ratio')
	parser.add_argument('--max-iter', type=int, default=500, metavar='LR', help='number of iterations (default: 500)')
	parser.add_argument('--pool-size', type=int, default=500, metavar='M', help='size of training examples pool (default: 500)')
	parser.add_argument('--label', type=int, default=0, metavar='M', help='specific label to train, ordered from most common to least common (default: 0)')
	parser.add_argument('--method', type=int, default=0, metavar='N', help='specify routine to run (default: 0, which is a single run of main() with parser args')
	
	args = parser.parse_args()

	if args.method == 0:
		# set seed
		np.random.seed(args.seed)

		# run main
		main(args)

	else:
		# run main multiple times with different max_iter, and report mean + std accuracy vs. max_iter for Chebyshev and Gravity, and 500/1000 pool size 
		seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
		query_list = [1, 5, 10, 20, 30, 40]
		label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		# label_list = [0]  
		pool_list = [500, 1000]
		accuracy_list = np.zeros( (len(label_list), 2, len(pool_list), len(query_list)) )  # mean accuracy 
		std_list = np.zeros( (len(label_list), 2, len(pool_list), len(query_list)) )  # std accuracy 
		cut_list = np.zeros( (len(label_list), 2, len(pool_list), len(query_list)) )  # mean number of cuts

		for i in range(len(label_list)):  # classifier for each label 
			print("Classifier # ", i)
			for j in range(2):  # center option (Chebyshev or Gravity)
				for k, pool_size in enumerate(pool_list, 0):  # pool size (500 or 1000)
					for l, n_query in enumerate(query_list, 0):  # query option
						accuracy_tmp = []
						cut_tmp = []
						for seed_val in seed_list:  # seed option
							np.random.seed(seed_val)

							# set arguments
							args.label = label_list[i]
							args.center = j
							args.pool_size = pool_size
							args.query = 0  # simple query as default 
							args.max_iter = n_query

							accuracy, n_cuts = main(args)
							accuracy_tmp.append(accuracy)
							cut_tmp.append(n_cuts)

						# compute average values over seeds
						mean_accuracy = np.mean(accuracy_tmp)
						std_accuracy = np.std(accuracy_tmp)
						mean_cuts = np.mean(cut_tmp)

						# store data 
						accuracy_list[i, j, k, l] = mean_accuracy
						std_list[i, j, k, l] = std_accuracy
						cut_list[i, j, k, l] = mean_cuts

		pdb.set_trace()
		np.save('accuracy', accuracy_list)
		np.save('std', std_list)
		np.save('cut', cut_list)



