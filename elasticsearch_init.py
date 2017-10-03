#!/usr/bin/env python3
import json
import os
import sys

from elasticsearch import Elasticsearch

from visimil.config import ELASTICSEARCH_HOSTS


if __name__ == "__main__":
    es = Elasticsearch(ELASTICSEARCH_HOSTS)

    request_body = \
        {'mappings':
            {'image':
                {'properties':
                    {"F" + str(f): {"type": "double"} for f in range(512)}}}}

    print("Initializing elasticsearch")
    es.indices.create(index='visimil', body=request_body, ignore=400)
