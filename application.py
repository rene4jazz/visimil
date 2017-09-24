#!/usr/bin/env python3
from __future__ import division
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import os

from io import BytesIO
from PIL import Image
import numpy as np
from numpy import linalg as LA
import requests
from flask import Flask, jsonify
from flask import request, make_response, abort
from elasticsearch import Elasticsearch


if os.environ.get('ES_HOSTS'):
    ELASTICSEARCH_HOSTS = \
        [{'host': es_host, 'port': 9200}
         for es_host in os.environ.get('ES_HOSTS').split(",")]
else:
    ELASTICSEARCH_HOSTS = None


app = Flask(__name__)
es = Elasticsearch(ELASTICSEARCH_HOSTS)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def get_features(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    target_size = (224, 224)
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')

    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x).flatten()
    return features.tolist()


def cos_similarity(a_vect, b_vect):
    return np.dot(a_vect, b_vect)/(LA.norm(a_vect, 2) * LA.norm(b_vect, 2))


@app.route('/api/v1/search', methods=['POST'])
def search():
    if not request.json or 'url' not in request.json:
        abort(400)

    features = get_features(request.json['url'])
    # np_features = np.asarray(features);

    acc = 0.2
    dim = 100
    if 'template' in request.json:
        if 'almost_identical' == request.json['template']:
            acc = 200
            dim = 500

    if 'accuracy' in request.json:
        acc = request.json['accuracy']
        if acc > 2000.0:
            acc = 2000.0
        if acc < 0.001:
            acc = 0.001
    offset = 1.0/acc**2
    if 'threshold' in request.json:
        dim = request.json['threshold']
        if dim > 512:
            dim = 512
        if dim < 1:
            dim = 1

    fields = []
    features_as_fields = []
    field_name = ''
    for i, f in enumerate(features):
        field_name = 'F'+str(i)
        fields.append(
            {'range': {field_name: {'gte': f-offset, 'lte': f+offset}}})
        features_as_fields.append({field_name: f})
    query = \
        {'query':
            {'bool':
                {'minimum_number_should_match': dim,
                 'should': fields}}}

    result = es.search(index='visimil', doc_type='image', body=query)

    results = []
    for hit in result['hits']['hits']:

        hit_f = \
            np.asarray(
                [list(v.values())[0] for v in sorted(
                    [{attr: value} for attr, value in hit['_source'].items()],
                key=lambda x:sorted(x.keys()))])

        cs = cos_similarity(features, hit_f)
        results.append({'id': hit['_id'], 'score': hit['_score'], 'cs': cs})

    return jsonify(
        {'accuracy': acc,
         'threshold': dim,
         'count': result['hits']['total'],
         'max_score': result['hits']['max_score'],
         'results': sorted(results, key=lambda k: k['cs'], reverse=True)})


@app.route('/api/v1/add', methods=['POST'])
def add_image():
    if not request.json or 'url' not in request.json or 'id' not in request.json:
        abort(400)

    features = get_features(request.json['url'])
    doc = {}
    for i, f in enumerate(features):
        doc["F" + str(i)] = f

    result = \
        es.index(
            index='visimil', id=request.json['id'], doc_type='image', body=doc)
    return jsonify({'result': result})


@app.route('/health_check', methods=['GET'])
def elasticsearch_check():
    app_health = \
        {'app_health': {
             'app_ok': False,
             'reasons': {'elasticsearch_ok': False}}}

    if not ELASTICSEARCH_HOSTS:
        return jsonify(app_health)
    else:
        app_health = app_health['app_health']['app_ok'] = True
        app_health = app_health['app_health']['reasons']['elasticsearch_ok'] = True
        return jsonify(app_health)


if __name__ == '__main__':
    app.run(debug=True)
