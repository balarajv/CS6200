import gensim
from collections import defaultdict
from collections import OrderedDict
from ElasticSearchHelper import *
from gensim.parsing.preprocessing import STOPWORDS
import os
import numpy as np

class topicIdentification(object):
    """docstring for topicIdentification"""
    def __init__(self):
        self.es = ES()
        self.queries = self.get_query_nos()
            
    def get_query_nos(self):
        with open("query_desc.short.txt") as input:
            lines = input.readlines()
        queries = []
        for line in lines:
            queries.append(line.strip().split()[0])
        return queries

    def read_qrel(self, doc_list):
        #51 0 AP890104-0259 0
        with open("qrels.adhoc.51-100.AP89.txt") as input:
            lines = input.readlines()

        for line in lines:
            query_id, _, doc_id, label = line.strip().split()
            if label == "1" and query_id in doc_list:
                doc_list[query_id].add(doc_id)
        return doc_list

    def read_BM25(self, doc_list):
        #56 Q0 AP890224-0245 1 27.263665620101182 Nil
        with open("okpitfbm25.txt") as input:
            lines = input.readlines()
        for line in lines:
            query_id,_,doc_id,_,_,_ = line.strip().split()
            doc_list[query_id].add(doc_id)
        return doc_list

    def get_texts(self):
                                queries = self.queries
                                doc_list = {}
                                for query in queries:
                                        doc_list[query] = set()
                                doc_list = self.read_qrel(doc_list)
                                doc_list = self.read_BM25(doc_list)
                                
                                texts = defaultdict(lambda:[])
                                for q_id in queries:
                                        doc_ids = list(doc_list[q_id])
                                        for doc_id in doc_ids:
                                                texts[q_id].append(self.tokenize(
                                                        self.es.get_doc_text(doc_id)))
                                return texts

    def get_dictionary(self, again, texts):

                for query in self.queries:
                    dictionary = gensim.corpora.Dictionary(texts[query])
                    dictionary.filter_extremes(no_below=5, no_above=0.8)
                return dictionary

    def get_corpus(self, again, texts, dictionary):
        if not again and os.path.isfile("part1_corpus.mm"):
            corpus = gensim.corpora.MmCorpus("part1_corpus.mm")
            return corpus
        corpus = [dictionary.doc2bow(text) for text in texts]
        gensim.corpora.MmCorpus.serialize("part1_corpus.mm", corpus)
        return corpus

    def do_LDA(self, again = False):
        if(not again and os.path.isfile("lda.model")):
       	    return gensim.models.LdaModel.load("lda.model")

        texts = []
        if (again or
        	(not os.path.isfile("part1_dictionary.dict") and
            not os.path.isfile("part1_corpus.mm"))):
            texts = self.get_texts()

        dictionary = self.get_dictionary(again, texts)
        corpus = self.get_corpus(again ,texts, dictionary)

        model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20,passes=5, alpha='auto')
        model.save("lda.model")
        return model

    def load_LDA(self):
    	return gensim.models.LdaModel.load("lda.model")

    def tokenize(self, text):
        l_text = text.lower()
        return [token for token in l_text.strip().split() if token not in STOPWORDS]

    def form_documents(self):
        pass


if __name__ == '__main__':
    ti = topicIdentification()
    lda_model = ti.load_LDA()
    for i in lda_model.print_topics(20):
    	print i

    """
    #model.printTopics(10)
    print model.print_topics(10)
    print "======================"
    print model.show_topic(19)
    topics_matrix = model.show_topics(formatted=False, num_words=20)
    print topics_matrix
    print len(topics_matrix)
    topics_matrix = np.array(topics_matrix)
    print "--------------------------------------"
    print "--------------------------------------"
    print "--------------------------------------"
    topic_words = topics_matrix[:,:,1]
    for i in topic_words:
        print([str(word) for word in i])
        print()
    #print model.print_topics(-1)
    """
