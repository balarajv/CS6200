from elasticsearch import Elasticsearch
from elasticsearch import helpers
from cleanStuffYo import *
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class ES(object):
        """docstring for ElasticSearchHelper"""
        def __init__(self, indexName, docType):
            self.instace = Elasticsearch(timeout=100)
            self.indexName = indexName
            self.docType   = docType
            if self.instace.indices.exists(indexName):
                  return  
            
            if(indexName == None or docType == None):
                   raise ValueError("indexName and docType can not be empty")

            request_body = {
                "settings": {
                    "index": {
                        "store": {
                            "type": "default"
                        },
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
            } 
            self.records = {}
            self.instace.indices.create(index=indexName, body=request_body)
            self.instace.indices.put_mapping(
                index=indexName,
                doc_type=docType,
                body={
                    docType: {
                        'properties': {
                            'from': {'type': 'string',
                                     'store': True,
                                     'index': 'analyzed',
                                     'term_vector': 'with_positions_offsets_payloads',
                                     },
                            'to': {'type': 'string',
                                       'store': True},
                            'subject': { 'type': 'string',
                                         'store': True,
                                         'index': 'analyzed',
                                         'term_vector': 'with_positions_offsets_payloads'},

                            'content': { 'type': 'string',
                                                 'store': True,
                                         'index': 'analyzed',
                                         'term_vector': 'with_positions_offsets_payloads'},
                            'label': {'type': 'string'},
                            'split': {'type': 'string'}
                        }
                    }
                })

        def add_to_elastic_search(self, mails):
                #def __init__(self, file, to, from1, subject, content, split, label):
                index = []
                count = 0
                for mail in mails:
                        print count
                        count += 1
                        record = {}
                        record["from"] = mail.from1
                        record["to"] = mail.to
                        record["subject"] = mail.subject.encode('ascii', 'ignore').decode('ascii')
                        record["content"] = mail.content.encode('ascii', 'ignore').decode('ascii')
                        record["label"] = mail.label
                        record["split"] = mail.split
                        index.append({'index': {'_index': self.indexName, '_type': self.docType, '_id': mail.file}})    
                        index.append(record)
                        if(count %10000 == 0):  
                            self.instace.bulk(index=self.indexName, body=index)
                            index = []
                if(len(index) != 0):
                    self.instace.bulk(index=self.indexName, body=index)

        def get_term_vectors(self, file):
                print "file ---------",
                print file
                try:
                    term_vectors = self.instace.termvectors(index = self.indexName, doc_type = self.docType, id=file, field_statistics = False,
                        offsets = False, positions = False, payloads = False )
                    return term_vectors["term_vectors"]["content"]["terms"]
                except Exception, e:
                    print "error"
                    return {}

        def get_file_list(self, type):
            files = []
            query = {
                        "query" : { 
                            "match" : 
                                { "split" : type}
                        },
                        "fields": ["_id"],
                        "size":10
                    }
            response = helpers.scan(client=self.instace, query=query, index=self.indexName, doc_type=self.docType)
            for hit in response:
                files.append(hit["_id"])
            return files

        def get_data(self, type):
                if type == "train":
                        pass
                elif type == "test":
                        pass


if __name__ == '__main__':
    es = ES("ass7", "data")
    print es.get_term_vectors("inmail.1")

 