import tensorflow as tf

from rotate_cert import detect_face
from east_part import locality_aware_nms as nms_locality
from east_part import lanms
from east_part.east_segment_line import resize_image, detect, sort_poly

import cv2
import time
import math
import os
import numpy as np
from keras.models import model_from_json

from east_part import model
from east_part.icdar import restore_rectangle

# for sort east cutted lines
from east_part.sort_cut_line_v2 import process_sort_cut_line

class id_card_model():
    def __init__(self):

        self._rotate_part = self._load_rotate_part()
        print('MTCNN model loaded!')
        self._east_part = self._load_east_part('./east_part/east_model')
        print('EAST model loaded!')
        self._keras_graph, self._number_model, self._other_model = None, None, None
        self._load_keras_models()
        print('Keras models loaded!')

    @property
    def mtcnn_models(self):
        return self._rotate_part

    def _load_rotate_part(self):
        mtcnn_graph = tf.Graph()

        with mtcnn_graph.as_default():
            mtcnn_sess = tf.Session()
            return detect_face.create_mtcnn(mtcnn_sess, None)

    def _load_east_part(self, model_path):
        east_graph = tf.Graph()

        with east_graph.as_default():
            input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            f_score, f_geometry = model.model(input_images, is_training=False)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            # ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            ckpt_state = tf.train.get_checkpoint_state(model_path)
            # model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            model_path = os.path.join(model_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            self._east_sess = sess
            self._f_score = f_score
            self._f_geometry = f_geometry
            self._input_images = input_images

    def east_predict(self, src_image):
        # convert RGB
        im = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = resize_image(im)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        score, geometry = self._east_sess.run([self._f_score, self._f_geometry], feed_dict={self._input_images: [im_resized]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        print('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        line_box = []
        im_with_box = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # save to file
        if boxes is not None:
            for box in boxes:
                # to avoid submitting errors
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                line_box.append(
                    [box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]])
                # f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                #     box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                # ))
                cv2.polylines(im_with_box, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                              color=(255, 255, 0), thickness=1)

        line_box_img = process_sort_cut_line(src_image, line_box)

        return im_with_box, line_box, line_box_img

    def _load_keras_models(self):
        self._keras_graph = tf.Graph()

        with self._keras_graph.as_default():
            self._number_model = model_from_json(open('number_recognition/model_structure_number.json').read())
            self._number_model.load_weights('number_recognition/model_weight_number.h5')
            self._number_model._make_predict_function()

            self._other_model = model_from_json(open('other_recognize/combined_model_structure.json').read())
            self._other_model.load_weights('other_recognize/combined_model_weight.h5')
            self._other_model._make_predict_function()

    def number_predict(self, data):

        with self._keras_graph.as_default():

            predict_class = self._number_model.predict(data, batch_size=18, verbose=0)
            # predict_class = model.predict_classes(data, batch_size=batch_size,verbose=0)
            predict_list = np.argmax(predict_class, axis=1).tolist()
            # predict_list = predict_class.tolist()
            # 如果识别的是数字，类型10对应数字 'X'
            result = ['X' if i == '10' else i for i in predict_list]

        return result

    def other_predict_prob(self, data):

        with self._keras_graph.as_default():
            prob = self._other_model.predict(data, batch_size=64, verbose=0)
            predict_class = np.argmax(prob, axis=1)

        return predict_class, prob.tolist()
    def other_predict_class(self, data, mappingList):

        with self._keras_graph.as_default():
            predict_class = self._other_model.predict_classes(data, batch_size=64, verbose=0)
            predict_list = predict_class.tolist()
            result = [mappingList[i] for i in predict_list]

        return result

if __name__ == '__main__':
    a = id_card_model()
