import logging
from PIL import Image
from io import BytesIO
import datetime
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tarfile
import os

def rotate(jpg_file):
    img=jpg_file.split("/")[-1]
    
    
    if '_100_' in img or '_101_' in img or '_102_' in img:
        image = cv2.rotate(jpg_file, cv2.ROTATE_90_CLOCKWISE)
    elif '_200_' in img or '_201_' in img or '_202_' in img:
        image = cv2.rotate(jpg_file, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return image

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

       ## 
    #     self.graph = tf.Graph()

    #     graph_def = None
    #     graph_def = tf.compat.v1.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read()) 

    #     if graph_def is None:
    #         raise RuntimeError('Cannot find inference graph in tar archive.')

    #     with self.graph.as_default():
    #         tf.import_graph_def(graph_def, name='')

    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    # #    session = tf.Session(config=config, ...)

    #     self.sess = tf.Session(graph=self.graph, config=config)
    ##

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


def apply_segmentation(image, seg_path,model):

    # get path and generate output path from it
    
    resized_im, seg_map = model.run(image)#orignal_im)


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
    img = img.convert('RGB').resize(image.size, Image.ANTIALIAS)
    img.save('./output.png')

    logging.info("saved file to" + seg_path)
    img.save(seg_path)

    return seg_path