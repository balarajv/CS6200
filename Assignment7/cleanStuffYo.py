from os import listdir
from bs4 import BeautifulSoup
import email
import random
from ElasticSearchHelper import *
from collections import defaultdict
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
import traceback
import pickle
import os.path
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class Mail(object):
    """docstring for Mails"""
    """
    unfortunately from is taken so from1 dumb name
    """
    def __init__(self, file, to, from1, subject, content, split, label):
    	self.file = file
        self.from1 = from1
        self.to = to
        self.subject = subject
        self.content = content
        self.split = split
        self.label = label


class CleanStuffYo(object):
    def __init__(self):
        self.es = ES("ass7","data")

    def read_from_files(self):
        mails = []
        if (os.path.exists("all_mails.pkl")):
        	with open('all_mails.pkl', 'rb') as input:
                	mails = pickle.load(input)
            	return mails

        labels = self.assign_labels()
        print "done labels"
        splits = self.assign_splits(labels)
        print "done splits"
        count = 0
        for file in listdir("trec07p/data/"):
            count += 1
            print count
            with open("trec07p/data/"+file, 'r') as input:
            	content = str = unicode(input.read(), errors='ignore')
                data = email.message_from_string(content)
                subject = data["subject"]
                if(subject == None):
                	subject = ""
                mails.append(Mail( file,
                                    data["to"],
                                    data["from"],
                                    subject,
                                    (subject +" "+self.clean_text(data)),
                                    "train" if splits[file] else "test",
                                    labels[file]))
        print "end of for loop"
        with open('all_mails.pkl', 'wb') as output:
        	pickle.dump(mails, output, pickle.HIGHEST_PROTOCOL)
        return mails

    def add_to_elastic_search(self, mails):
        self.es.add_to_elastic_search(mails)

    """Change the below funtion if it needs further processing"""
    def clean_text(self, data):
        clean_text = ""
        if("Content-Type" in data and "image" in data["Content-Type"]):
            print "it is an image"
            return ""
        try:
            if data.is_multipart():
                clean_text = self.extract_from_multi_part(data)
            else:
                clean_text = self.extract_from_soup(data.get_payload())
        except Exception, e:
            print "error "
            print traceback.print_exc()
            raise e
            return ""
        return clean_text

    def extract_from_multi_part(self, data):
            clean_text = ""
            for payload in data.get_payload():
                    if "Content-Type" in payload and "image" in payload["Content-Type"]:
                            continue
                    if payload.is_multipart():
                            clean_text += self.extract_from_multi_part(payload)
                    else:
                            clean_text += self.extract_from_soup(payload.get_payload())
            return clean_text

    def extract_from_soup(self, text12):
    	#print "here is the error "
        soup = BeautifulSoup(text12, "html.parser")
        for script in soup(["script", "style"]):
                script.extract()  # rip it out          
        return soup.get_text()

    def assign_splits(self, labels):
        splits = defaultdict(lambda: True)
        """20% of 75419 is 15084"""
        test = random.sample(xrange(1, 75419), 15084)
        count = 1
        for label in labels:
            if count in test:
                splits[label] = False
            else:
                splits[label] = True
            count += 1
        return splits

    def assign_labels(self):
        labels = {}
        with open("trec07p/full/index") as doc:
            lines = doc.readlines()
        for line in lines:
            label, file = line.strip().split()
            file = file.replace("../data/", "")
            labels[file] = label
        return labels

    

if __name__ == '__main__':
    clean = CleanStuffYo()
    mails = clean.read_from_files()
    print len(mails)
    clean.add_to_elastic_search(mails)
