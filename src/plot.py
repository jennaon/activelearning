import numpy as np  
import matplotlib.pyplot as plt 

def main():
	# load data 
	accuracy = np.load('accuracy.npy')
	std = np.load('std.npy')
	n_cuts = np.load('cut.npy')

	# dimension
	m, n, o, p = np.shape(accuracy)  # (label index, center option, pool size, query number)

	# transfer test parameters
	seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
	query_list = [1, 5, 10, 20, 30, 40]
	label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	# label_list = [0]  
	pool_list = [500, 1000]

	# print(std[0, 0, 0, 0])

	# plot accuracy vs. queries for each center and pool option on the same plot   
	plt.figure()
	plt.errorbar(query_list, accuracy[0, 0, 0, :], std[0, 0, 0, :], linestyle='-', marker='.', capsize=3, markersize=10)
	plt.errorbar(query_list, accuracy[0, 0, 1, :], std[0, 0, 1, :], linestyle='-', marker='.', capsize=3, markersize=10)
	plt.errorbar(query_list, accuracy[0, 1, 0, :], std[0, 1, 0, :], linestyle='-', marker='.', capsize=3, markersize=10)
	plt.errorbar(query_list, accuracy[0, 1, 1, :], std[0, 1, 1, :], linestyle='-', marker='.', capsize=3, markersize=10)
	plt.xlabel('Queries')
	plt.ylabel('Accuracy')
	plt.legend(['Chebyshev/500', 'Chebyshev/1000', 'Gravity/500', 'Gravity/1000'])
	plt.ylim([0.4, 1])
	plt.grid()
	plt.show()

	# plot number of cuts per query
	plt.figure()
	plt.plot(query_list, n_cuts[0, 0, 0, :], linestyle='-', marker='.', markersize=10)
	plt.plot(query_list, n_cuts[0, 0, 1, :], linestyle='-', marker='.', markersize=10)
	plt.plot(query_list, n_cuts[0, 1, 0, :], linestyle='-', marker='.', markersize=10)
	plt.plot(query_list, n_cuts[0, 1, 1, :], linestyle='-', marker='.', markersize=10)
	plt.xlabel('Queries')
	plt.ylabel('Number of cuts')
	plt.legend(['Chebyshev/500', 'Chebyshev/1000', 'Gravity/500', 'Gravity/1000'])
	plt.ylim([0, 10])
	plt.grid()
	plt.show()


if __name__ == '__main__':
	main()