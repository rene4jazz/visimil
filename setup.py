import json

from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

flist = list(range(512))

fields = {}

for f in flist:
     fields["F" + str(f)] = {"type": "double"} 

request_body = {'mappings': {'image': {'properties': fields}}}

es.indices.create(index='visimil', body = request_body, ignore=400)

