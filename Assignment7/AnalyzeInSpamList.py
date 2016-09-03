from ElasticSearchHelper import ES
import sys
sys.path.insert(0, "/home/bala/Desktop/IR assignment/Assignment7/liblinear-master/python")
from liblinear import *
import numpy
import pickle
from collections import defaultdict
from collections import OrderedDict
from sklearn import linear_model
from scipy.sparse import coo_matrix
from sklearn.cross_validation import cross_val_predict
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import operator

class PredictDetail(object):
	 	"""docstring for ClassName"""
	 	def __init__(self, file_name, spam_prob, ham_prob, score):
	 		self.file_name = file_name
	 		self.spam_prob = spam_prob
	 		self.ham_prob  = ham_prob
	 		self.score = score
	 	def __str__(self):
	 		print self.file_name +" "+ self.spam_prob+" "+self.ham_prob+" "+score


class InSpamList(object):
	"""docstring for InSpamList"""
	def __init__(self):
		self.es = ES("ass7", "data")
		self.term_number = {}

	def get_spam_list(self):
		with open("trec07p/spam_words_new") as input:
			spam_list = input.read().splitlines() 
		return list(set(spam_list))

	def get_label_list(self):
		labels = {}
		files = []
		with open("trec07p/full/index") as doc:
			lines = doc.readlines()
		for line in lines:
			label, file = line.strip().split()
			file = file.replace("../data/", "")
			files.append(file)
			labels[file] = 1 if label == "spam" else 0

		return files, labels

	def get_term_vectors(self, file_list):
		"""Cant pickle with lambda so manully loop initailize"""
		term_vectors = {}

		for file in file_list:
			term_vectors[file] = {}
		
		if (os.path.exists("term_vectors.pkl")):
			with open('term_vectors.pkl', 'rb') as input:
				term_vectors = pickle.load(input)
				return term_vectors

		for file in file_list:
			term_details = self.es.get_term_vectors(file)
			for term in term_details:
				term_vectors[file][term] = int(term_details[term]["term_freq"])


		with open('term_vectors.pkl', 'wb') as output:
			pickle.dump(term_vectors, output, pickle.HIGHEST_PROTOCOL)
		return term_vectors

	def get_integer_value(self, term):
		if not term in self.term_number:
			self.term_number[term] = len(self.term_number)

		return self.term_number[term]


	def get_train_test_data(self):
		train_files, test_files = self.es.get_train_test_data()


	def train_and_test_dense(self, spam_list, file_list, train_file_list, test_file_list, labels):
		
		term_details   = self.get_term_details(spam_list)

		matrix_train_x, matrix_train_y = self.form_matrix_dense(train_file_list, spam_list, term_details, labels)
		matrix_test_x, matrix_test_y   = self.form_matrix_dense(test_file_list, spam_list, term_details, labels)

		""" Using logistic regression as """
		#lr = linear_model.LogisticRegression()
		lr = tree.DecisionTreeClassifier()
		lr.fit(matrix_train_x, matrix_train_y)
		#scores = lr.coef_
		#coeffs = {}
		#count = 0
		#print spam_list[0:20]
		#print scores
		#for i in range(0, len(spam_list), 1):
		#	coeffs[spam_list[i]] = scores[0][i]

		#sorted_x = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)
		#print sorted_x


		matrix_train_predict_y = lr.predict(matrix_train_x)
		matrix_test_predict_y  = lr.predict(matrix_test_x)
		print "Test Sparse accuracy ",
		print self.check_percentage(matrix_test_y, matrix_test_predict_y)
		print "Train Sparce accuracy ",
		print self.check_percentage(matrix_train_y, matrix_train_predict_y)
		
		test_predict  = lr.predict_proba(matrix_test_x)
		train_predict = lr.predict_proba(matrix_train_x)
			
		train_predict_details = self.get_predict_details(train_file_list, train_predict, labels)
		test_predict_details  = self.get_predict_details(test_file_list, test_predict, labels)

		#print type(matrix_test_x_s)
		#print type(matrix_test_y_s)

		#matrix_train_predict_y = cross_val_predict(lr, matrix_train_x, matrix_train_y, cv=10)
		#matrix_test_predict_y = cross_val_predict(lr, matrix_test_x, matrix_test_y, cv=10)

		#print self.check_percentage(matrix_train_y_s, matrix_train_predict_y)
		#print self.check_percentage(matrix_test_y_s, matrix_test_predict_y)

		self.write_to_file(train_predict_details, "train_dense_633")
		self.write_to_file(test_predict_details, "test_dense_633")

	def check_percentage(self, matrix_y, matrix_predict_y):
		count = 0
		for i in range(0, len(matrix_y), 1):
			if(matrix_y[i] == matrix_predict_y[i]):
				count += 1
		return float(count)/len(matrix_y)

	def train_and_test_sparse(self, file_list, train_file_list, test_file_list, labels):

		term_vectors   = self.get_term_vectors(file_list)

		matrix_train_x, matrix_train_y = self.form_matrix_sparse(train_file_list, labels, term_vectors)
		matrix_test_x, matrix_test_y   = self.form_matrix_sparse(test_file_list, labels, term_vectors)

		""" Using logistic regression as """
		lr = linear_model.LogisticRegression()
		lr.fit(matrix_train_x, matrix_train_y)
		
		#scores = lr.coef_
		#coeffs = {}
		#count = 0
		#term_spam_list = []
		#term_indexes = self.create_term_indexes(file_list, term_vectors)

		#sorted_term_indexes = sorted(term_indexes.items(), key=operator.itemgetter(1))
		#print sorted_term_indexes[0:20]
		#for i in range(0, len(sorted_term_indexes), 1):
		#	coeffs[sorted_term_indexes[i][0]] = scores[i]

		#sorted_x = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)
		#print sorted_x[0:50]

		matrix_test_predict_y   = lr.predict(matrix_test_x)
		matrix_train_predict_y  = lr.predict(matrix_train_x)
		print "test dense accuracy ",
		print self.check_percentage(matrix_test_y, matrix_test_predict_y)

		print "train dense accuracy ",
		print self.check_percentage(matrix_train_y, matrix_train_predict_y)

		test_predict  = lr.predict_proba(matrix_test_x)
		train_predict = lr.predict_proba(matrix_train_x)

		train_predict_details = self.get_predict_details(train_file_list, train_predict, labels)
		test_predict_details  = self.get_predict_details(test_file_list, test_predict, labels)

		#matrix_train_predict_y = cross_val_predict(lr, matrix_train_x_s, matrix_train_y_s, cv=10)
		#matrix_test_predict_y = cross_val_predict(lr, matrix_test_x_s, matrix_test_y_s, cv=10)

		#print self.check_percentage(matrix_train_y_s, matrix_train_predict_y)
		#print self.check_percentage(matrix_test_y_s, matrix_test_predict_y)

		print "writing into file"
		self.write_to_file(train_predict_details, "train_sparse_633")
		self.write_to_file(test_predict_details, "test_sparse_633")


	def create_indexes(self, some_list):
		indexes = {}
		count = 0
		for element in some_list:
			indexes[element] = count 
			count += 1
		return indexes

	def create_term_indexes(self, file_list, term_vectors):
		indexes = {}
		count = 0
		if (os.path.exists("term_indexes.pkl")):
			with open('term_indexes.pkl', 'rb') as input:
				indexes = pickle.load(input)
				return indexes

		for fil in term_vectors:
			for term in term_vectors[fil]:
				if term not in indexes:
					indexes[term] = count
					count += 1
		
		with open('term_indexes.pkl', 'wb') as output:
			pickle.dump(indexes, output, pickle.HIGHEST_PROTOCOL)

		return indexes

	def form_matrix_dense(self, file_list, spam_list, term_details, labels):
		no_rows    = len(file_list)
		no_columns = len(spam_list)
		matrix_x  = numpy.zeros(shape=(no_rows, no_columns))
		matrix_y  = numpy.zeros(no_rows)

		no_rows = 0
		for file in file_list:
			row = []
			for term in spam_list:
				if term in term_details and file in term_details[term]:	
					row.append(term_details[term][file])
				else:
				    row.append(0)
			matrix_x[no_rows] = row
			matrix_y[no_rows] = labels[file]
			no_rows += 1
		return matrix_x, matrix_y

	def form_matrix_sparse(self, file_list, labels, term_vectors):
		
		file_indexes = self.create_indexes(file_list)
		term_indexes = self.create_term_indexes(file_list, term_vectors)
		
		no_rows    = len(file_list)
		no_columns = len(term_indexes)

		row    = []
		column = []
		data   = []

		matrix_y  = numpy.zeros(no_rows)

		for fil in file_list:
			term_vector = term_vectors[fil]
			for term in term_vector:
				row.append(file_indexes[fil])
				column.append(term_indexes[term])
				data.append(term_vector[term])
				#data.append(1)
			matrix_y[file_indexes[fil]] = labels[fil]

		matrix_x = coo_matrix((numpy.array(data), (numpy.array(row), numpy.array(column))), shape=(no_rows, no_columns))
		return matrix_x, matrix_y

	def get_predict_details(self, file_list, predict, labels):
		predict_details = []
		count = 0
		for file in file_list:
			predict_details.append(PredictDetail(file, predict[count][1], predict[count][0], labels[file]))
			count += 1
		predict_details.sort(key=lambda x: x.spam_prob, reverse=True)
		return predict_details

		
	def write_to_file(self, predict_details, file_name):
		f = open(file_name, "w")
		for predict_detail in predict_details:
			f.write(predict_detail.file_name+" "+str(predict_detail.spam_prob)+" "+str(predict_detail.ham_prob)+" "+str(predict_detail.score))
			f.write("\n")
		f.close()

	def get_term_details(self, spam_list):
		term_details = {}

		for term in spam_list:
			term_details[term.strip()] = self.es.get_details_for_term(term.strip())
		return term_details

	def get_feature_value_sparse(words, term_vector):
		words = words.strip().split()
		score = 0
		for word in words:
			if word in term_vector:
				score += term_vector[word]
		return score


if __name__ == '__main__':
	analyze = InSpamList()
	spam_list = analyze.get_spam_list()
	print type(spam_list)

	file_list, labels = analyze.get_label_list()

	#analyze.get_term_details(spam_list)	

	train_file_list = analyze.es.get_file_list(type="train")
	test_file_list  = analyze.es.get_file_list(type="test")

	analyze.train_and_test_dense(spam_list, file_list, train_file_list, test_file_list, labels)
	analyze.train_and_test_sparse(file_list, train_file_list, test_file_list, labels)
	
	#spam_list, file_list, train_file_list, test_file_list, labels
