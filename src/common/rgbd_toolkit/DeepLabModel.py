import logging
from PIL import Image
from io import BytesIO
import datetime
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tarfile
import os


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        # self.graph = tf.Graph()

        # graph_def = None
        # # Extract frozen graph from tar archive.
        # tar_file = tarfile.open(tarball_path)
        # for tar_info in tar_file.getmembers():
        #     if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        #         file_handle = tar_file.extractfile(tar_info)
        #         graph_def = tf.GraphDef.FromString(file_handle.read())
        #         break

        # tar_file.close()

        # if graph_def is None:
        #     raise RuntimeError('Cannot find inference graph in tar archive.')

        # with self.graph.as_default():
        #     tf.import_graph_def(graph_def, name='')

        #self.sess = tf.Session(graph=self.graph)

        
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.compat.v1.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read()) 

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    #    session = tf.Session(config=config, ...)

        self.sess = tf.Session(graph=self.graph, config=config)

    def run(self, image):
        """Runs inference on a single image.
        Args:
          image: A PIL.Image object, raw input image.
        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        start = datetime.datetime.now()

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        logging.info("Time taken to evaluate segmentation is : " + str(diff))

        return resized_image, seg_map


def apply_segmentation(jpg_path, seg_path,model):

    # get path and generate output path from it
    seg_path = jpg_path.replace(".jpg", "_SEG.png")

    # load image from path
    try:
        logging.info("Trying to open : " + jpg_path)
        jpeg_str   = open(jpg_path, "rb").read()
        orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        logging.error('Cannot retrieve image. Please check file: ' + jpg_path)
        return


    # apply segmentation via pre trained model
    logging.info('running deeplab on image %s...' % jpg_path)
    resized_im, seg_map = model.run(orignal_im)


    # convert the image into a binary mask
    width, height = resized_im.size
    dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = seg_map[y,x]
            if color == 0:
                dummyImg[y,x] = [0, 0, 0, 255]
            else :
                dummyImg[y,x] = [255,255,255,255]

                
    img = Image.fromarray(dummyImg)
    img = img.convert('RGB').resize(orignal_im.size, Image.ANTIALIAS)
    img.save('./output.png')

    logging.info("saved file to" + seg_path)
    img.save(seg_path)

    return seg_path