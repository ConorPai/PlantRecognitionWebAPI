#coding=utf-8

import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

import tensorflow as tf
from flask import Flask, request
app = Flask(__name__)

#加载识别标签翻译字典
dictionary = {}
ftotal = open('./dictionarys.txt', 'r')
line = ftotal.readline()
while line:
    totaltree = line.replace('\r\n', '').replace('\n', '').split(',')
    dictionary[totaltree[0]] = totaltree[1].decode("utf-8")
    line = ftotal.readline()
ftotal.close()

#加载识别标签
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

        #构建图片文件存储路径
        photofile = '/Users/paiconor/Downloads/uploadPhoto/' + file.filename

        #保存图片文件
        file.save(photofile)

        #返回植物图片识别结果
        return plantRecognition(photofile), 200

def plantRecognition(plantFile):

    #打开植物图片进行识别
    image = tf.gfile.FastGFile(plantFile, 'rb').read()
    predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})

    #根据分类概率进行排序
    top = predict[0].argsort()[-len(predict[0]):][::-1]

    result = ''
    count = 8
    for index in top:
        human_string = dictionary[labels[index]].decode("utf-8")
        score = predict[0][index]

        #识别率低于0.01即跳出
        if score < 0.0099:
            break

        #拼写识别结果
        temp = human_string + ':' + '%(p).2f'%{'p':score * 100} + '%'
        result += temp.decode("utf-8") + '\n'

        #只返回识别前8个结果
        count -= 1
        if count == 0:
            break

    print result
    return result


if __name__ == '__main__':
    app.run(host='192.168.1.101', port=5000)
