#coding=utf-8

import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

import cv2
import math
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
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

        GaussianBlur(photofile)

        #返回植物图片识别结果
        return plantRecognition(photofile), 200

#植物识别，返回识别结果字符串
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

#图片按中心点旋转
def rotate_about_center(src, angle, scale=1.):
    if angle == 0:
        return src

    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

#图片高斯模糊处理
def GaussianBlur(imagefile):

    rotateangle = get_Rotate_Angle(imagefile)

    img = cv2.imread(imagefile)
    kernel_size = (5, 5)
    sigma = 3
    newimg = cv2.GaussianBlur(img, kernel_size, sigma)
    res = rotate_about_center(newimg, rotateangle)

    cv2.imwrite(imagefile, res);

#获取图片旋转角度
def get_Rotate_Angle(fname):
    try:
        img = Image.open(fname)
        if hasattr(img, '_getexif'):
            exifinfo = img._getexif()
            if exifinfo != None:
                for tag, value in exifinfo.items():
                    decoded = TAGS.get(tag, tag)

                    if decoded == 'Orientation':

                        if value == 6:
                            return 270
                        else:
                            return 0
    except IOError:
        print 'IOERROR ' + fname
    return 0

if __name__ == '__main__':
    app.run(host='192.168.1.101', port=5000)
