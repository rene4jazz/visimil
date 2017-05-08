#!/usr/local/lib/python
from __future__ import division
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

from io import BytesIO
from PIL import Image
import numpy as np
import argparse, requests, json
from flask import Flask, jsonify
from flask import request, make_response, abort
from elasticsearch import Elasticsearch

app = Flask(__name__)

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def get_features(url):
  response = requests.get(url)
  img = Image.open(BytesIO(response.content)).convert('RGB')

  target_size=(224, 224)
  model = VGG16(weights='imagenet', include_top=False, pooling='max')

  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  features = model.predict(x).flatten()
  return features.tolist()

@app.route('/api/v1/search', methods=['POST'])
def search():
    if not request.json or not 'url' in request.json:
        abort(400)

    features = get_features(request.json['url'])


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

    fields  = []
    field_name = ''
    for i,f in enumerate(features):
         field_name = 'F'+str(i)
         fields.append({'range': {field_name: {'gte': f-offset, 'lte': f+offset} }})
    query = {'query': {'bool': {'minimum_number_should_match': dim,'should': fields}}}
 
    result = es.search(index='visimil',doc_type='image', body=query)
    results = []
    for hit in result['hits']['hits']:
        results.append({'id': hit['_id'], 'score': hit['_score']})

    return jsonify({'accuracy': acc, 'threshold': dim, 'count': result['hits']['total'], 'max_score': result['hits']['max_score'], 'results': results })

@app.route('/api/v1/add', methods=['POST'])
def add_image():
    if not request.json or not 'url' in request.json or not 'id' in request.json:
        abort(400)

    features = get_features(request.json['url'])
    
    doc  = {}
    for i,f in enumerate(features):
         doc["F" + str(i)] = f

    result = es.index(index='visimil', id=request.json['id'], doc_type='image', body=doc)
    return jsonify({'result': result})    

if __name__ == '__main__':
    app.run(debug=True)
