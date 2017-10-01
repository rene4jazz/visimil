#!/usr/bin/env python3
import json
import os
import sys

from elasticsearch import Elasticsearch

from visimil.config import ELASTICSEARCH_HOSTS


if __name__ == "__main__":
    es = Elasticsearch(ELASTICSEARCH_HOSTS)

    flist = range(512)
    fields = {}

    for f in flist:
        fields["F" + str(f)] = {"type": "double"}

    request_body = {'mappings': {'image': {'properties': fields}}}

    print("Initializing elasticsearch")
    es.indices.create(index='visimil', body=request_body, ignore=400)
