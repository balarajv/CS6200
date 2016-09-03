import pickle


with open("term_indexes.pkl") as input:
	x = pickle.load(input)

print len(x)
