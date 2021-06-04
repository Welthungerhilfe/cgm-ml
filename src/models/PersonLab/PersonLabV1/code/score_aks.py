import utils
import json
import numpy as np
import os
import logging
import logging.config
import sys
import time
from scipy.ndimage.filters import gaussian_filter
import cv2
import tensorflow as tf
import time

sys.path.append('/var/azureml-app/azureml-models/personlabV1/2/personlab')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

#tf.compat.v1.disable_eager_execution()


def init():
    print(time.time())
    MODEL_DIR = '/var/azureml-app/azureml-models/personlabV1/2/personlab/'
    multiscale = [1.0, 1.5, 2.0]

    global tf_img
    tf_img = []
    global outputs
    outputs = []
    for i in range(len(multiscale)):
        scale = multiscale[i]
        tf_img.append(tf.placeholder(tf.float32, shape=[1, int(scale * 401), int(scale * 401), 3]))
        outputs.append(utils.model(tf_img[i]))
    global sess
    sess = tf.Session()

    global_vars = tf.global_variables()
    saver = tf.train.Saver(var_list=global_vars)
    checkpoint_path = MODEL_DIR + 'model.ckpt'
    saver.restore(sess, checkpoint_path)
    logging.info(time.time())


def run(data):

    try:
        logging.info(time.time())
       #TODO find logger in
        multiscale = [1.0, 1.5, 2.0]
        batch_size, height, width = 1, 401, 401
        image_list = json.loads(data)
        pose_scoreslist = []
        pose_keypoint_scoreslist = []
        pose_keypoint_coordslist = []

        for i in range(1):
            if(i == 0):
                input_image = np.array(image_list['input_image1'], dtype=np.uint8)
            else:
                input_image = np.array(image_list['input_image2'], dtype=np.uint8)

            scale_outputs = []
            for i in range(len(multiscale)):
                scale = multiscale[i]
                cv_shape = (401, 401)
                cv_shape2 = (int(cv_shape[0] * scale), int(cv_shape[1] * scale))
                scale2 = cv_shape2[0] / 600
                input_img = cv2.resize(input_image, None, fx=scale2, fy=scale2)
                #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
                input_img = cv2.copyMakeBorder(
                    input_img,
                    0,
                    cv_shape2[0] -
                    input_img.shape[0],
                    0,
                    cv_shape2[1] -
                    input_img.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[
                        127,
                        127,
                        127])
                scale_img = input_img
                imgs_batch = np.zeros((batch_size, int(scale * height), int(scale * width), 3))
                imgs_batch[0] = scale_img

                one_scale_output = sess.run(outputs[i], feed_dict={tf_img[i]: imgs_batch})
                scale_outputs.append([o[0] for o in one_scale_output])

            sample_output = scale_outputs[0]
            for i in range(1, len(multiscale)):
                for j in range(len(sample_output)):
                    sample_output[j] += scale_outputs[i][j]
            for j in range(len(sample_output)):
                sample_output[j] /= len(multiscale)

            H = utils.compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])
            for i in range(17):
                H[:, :, i] = gaussian_filter(H[:, :, i], sigma=2)

            pred_kp = utils.get_keypoints(H)
            pred_skels = utils.group_skeletons(keypoints=pred_kp, mid_offsets=sample_output[2])
            pred_skels = [skel for skel in pred_skels if (skel[:, 2] > 0).sum() > 6]
            #logging.info('Number of detected skeletons: %d', len(pred_skels)

            pose_scores = np.zeros(len(pred_skels))
            pose_keypoint_scores = np.zeros((len(pred_skels), 17))
            pose_keypoint_coords = np.zeros((len(pred_skels), 17, 2))

            for j in range(len(pred_skels)):
                sum = 0
                for i in range(17):
                    sum += pred_skels[j][i][2] * 100
                    pose_keypoint_scores[j][i] = pred_skels[j][i][2] * 100
                    pose_keypoint_coords[j][i][0] = pred_skels[j][i][0]
                    pose_keypoint_coords[j][i][1] = pred_skels[j][i][1]
                pose_scores[j] = sum / 17

            pose_scoreslist.append(pose_scores)
            pose_keypoint_scoreslist.append(pose_keypoint_scores)
            pose_keypoint_coordslist.append(pose_keypoint_coords)

        result = json.dumps({'pose_scores': pose_scoreslist,
                             'keypoint_scores': pose_keypoint_scoreslist,
                             'keypoint_coords': pose_keypoint_coordslist})
        # You can return any data type, as long as it is JSON serializable.
        return result
    except Exception as e:
        error = str(e)
        return error
