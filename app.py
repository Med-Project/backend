import os
import tensorflow as tf
from flask import Flask, jsonify, request

from preprocess import prediction
from SiameseNetwork import SiameseNetwork

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

siamese = SiameseNetwork()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./siamese-model'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()['data']
    except Exception as e:
        responses = jsonify(error=404)
        responses.status_code = 404
        return responses

    s = data
    value = prediction(s, siamese)
    
    if(value == -1):
        responses = jsonify(error=404)
        responses.status_code = 404
        return responses
    
    res, siamese_res = value

    responses = jsonify(w2v=res, siamese=siamese_res)
    responses.status_code = 200

    return responses

if __name__ == '__main__':
     app.run()