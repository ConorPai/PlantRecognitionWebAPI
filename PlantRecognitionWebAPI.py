#coding=utf-8

import sys

import tensorflow as tf
import json
from flask import Flask, request, jsonify
app = Flask(__name__)

dictionary = {}
ftotal = open('./dictionarys.txt', 'r')
line = ftotal.readline()
while line:
    totaltree = line.replace('\r\n', '').replace('\n', '').split(',')
    dictionary[totaltree[0]] = totaltree[1].decode("utf-8")
    line = ftotal.readline()
ftotal.close()

labels = []
for label in tf.gfile.GFile("output_labels.txt"):
    labels.append(label.rstrip())

# 加载Graph
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

@app.route('/upload', methods=['GET', 'POST'])
def change_avatar():
    if request.method == 'POST':
        file = request.files['file']
        photofile = '/Users/paiconor/Downloads/uploadPhoto/' + file.filename
        file.save(photofile)

        return plantRecognition(photofile), 200

def plantRecognition(plantFile):

    image = tf.gfile.FastGFile(plantFile, 'rb').read()

    predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})

    # 根据分类概率进行排序
    top = predict[0].argsort()[-len(predict[0]):][::-1]

    result = ''
    count = 8
    for index in top:
        human_string = dictionary[labels[index]].decode("utf-8")
        score = predict[0][index]

        temp = human_string + ':' + '%(p).2f'%{'p':score * 100} + '%'
        result += temp.decode("utf-8") + '\n'

        count -= 1
        if count == 0:
            break

    print result
    return result


if __name__ == '__main__':

    default_encoding = 'utf-8'
    if sys.getdefaultencoding() != default_encoding:
        reload(sys)
        sys.setdefaultencoding(default_encoding)

    app.run(host='192.168.1.101', port=5000)
