#!/usr/bin/env python3
import json

from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])

flist = list(range(512))

fields = {}

for f in flist:
    fields["F" + str(f)] = {"type": "double"}

request_body = {'mappings': {'image': {'properties': fields}}}

es.indices.create(index='visimil', body=request_body, ignore=400)
