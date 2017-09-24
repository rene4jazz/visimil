#!/usr/bin/env python3
import json
import os
import sys

from elasticsearch import Elasticsearch


if __name__ == "__main__":
#    if os.environ.get('ES_HOSTS'):
#        ELASTICSEARCH_HOSTS = \
#            [{'host': es_host, 'port': 9200}
#             for es_host in os.environ.get('ES_HOSTS').split(",")]
#    else:
#        ELASTICSEARCH_HOSTS = None

#    try:
#        if ELASTICSEARCH_HOSTS:
#            es = Elasticsearch(ELASTICSEARCH_HOSTS)
#        else:
#            raise Exception
#    except Exception as e:
#        print(e)
#        sys.exit(1)
        
    ELASTICSEARCH_HOSTS = [{'host': "127.0.0.1", 'port': 9200}]
    es = Elasticsearch(ELASTICSEARCH_HOSTS)

    flist = range(512)
    fields = {}

    for f in flist:
        fields["F" + str(f)] = {"type": "double"}

    request_body = {'mappings': {'image': {'properties': fields}}}

    print("Initializing elasticsearch")
    es.indices.create(index='visimil', body=request_body, ignore=400)
