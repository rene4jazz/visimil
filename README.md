# visimil

Keras/Elasticsearch based visual similarity search (tested keras model using Tensorflow backend).

Visimil uses Keras VGG16 model with Imagenet pre-trained weights. It use last conv layer feature vector values to compare visual similarity on images. There is one endpoint to add new computed image features into elasticsearch and another for image lookup.   

## Dependencies

python modules.

* pillow
* requests
* numpy
* tensorflow
* keras
* h5py
* flask
* elasticsearch

# Setup
We need to create Elasticsearch mappings first. Run visimil_setup.py in order to setup mappings.
Edit visimil_setup.py and update Elasticsearch server and port values if needed on line #4


# Usage

run visimil.py and it will create a web server on port 5000

## Adding new entries to database.
HTTP POST to <hostname>/api/v1/add

 body payload
 
```
{
  "id": "Unique string",
  "url": "image url"
}
```


## searching for visually similar.
HTTP POST to <hostname>/api/v1/search

 body payload
 
```
{
  "url": "source image url",
  "accuracy": 0.2,
  "threshold": 100
}
```
### Optional values
Accuracy: offset values included as a hit for each dimension. (Values from  ``` 0 > x <=200 ```)
Threshold: minimun number of dimiension that need to match to score in result. (Values from 1 to 512)

