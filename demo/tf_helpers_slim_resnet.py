"""
    Helper functions to hide Tensorflow as Caffe.
"""
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim.nets import resnet_v1


import utils_resnet as utils

import matplotlib.pyplot as plt

NUM_OVER_SAMPLES = 10


def _oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.
    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((NUM_OVER_SAMPLES * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def _permute_params_to_caffe(data):
    if data.ndim == 4:
        data = np.transpose(data, (3, 2, 0, 1)) # (nodes, w, batch_size, h)
    elif data.ndim == 2:
        data = np.transpose(data)
    else:
        raise(ValueError, 'Well this is unexpected...')

    return data


def _permute_blobs_to_caffe(data):
    if data.ndim == 4:
        if data.ndim == 4:
            data = np.transpose(data, (0, 3, 1, 2)) # (batch_size, nodes, h, w)
        elif data.ndim == 2:
            data = np.transpose(data)
        else:
            raise (ValueError, 'Well this is unexpected...')

    return data



resnet_v1_101_caffe_to_tf = {
  'data':                 'images',
  'conv1':                'resnet_v1_101/conv1/Relu',
  'pool1':                'resnet_v1_101/pool1/MaxPool',
  'res2a_branch1':        'resnet_v1_101/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm',
  'res2a_branch2a':       'resnet_v1_101/block1/unit_1/bottleneck_v1/conv1/Relu',
  'res2a_branch2b':       'resnet_v1_101/block1/unit_1/bottleneck_v1/conv2/Relu',
  'res2a_branch2c':       'resnet_v1_101/block1/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res2a':                'resnet_v1_101/block1/unit_1/bottleneck_v1/Relu',
  'res2b_branch2a':       'resnet_v1_101/block1/unit_2/bottleneck_v1/conv1/Relu',
  'res2b_branch2b':       'resnet_v1_101/block1/unit_2/bottleneck_v1/conv2/Relu',
  'res2b_branch2c':       'resnet_v1_101/block1/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res2b':                'resnet_v1_101/block1/unit_2/bottleneck_v1/Relu',
  'res2c_branch2a':       'resnet_v1_101/block1/unit_3/bottleneck_v1/conv1/Relu',
  'res2c_branch2b':       'resnet_v1_101/block1/unit_3/bottleneck_v1/conv2/Relu',
  'res2c_branch2c':       'resnet_v1_101/block1/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res2c':                'resnet_v1_101/block1/unit_3/bottleneck_v1/Relu',
  'res3a_branch1':        'resnet_v1_101/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm',
  'res3a_branch2a':       'resnet_v1_101/block2/unit_1/bottleneck_v1/conv1/Relu',
  'res3a_branch2b':       'resnet_v1_101/block2/unit_1/bottleneck_v1/conv2/Relu',
  'res3a_branch2c':       'resnet_v1_101/block2/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res3a':                'resnet_v1_101/block2/unit_1/bottleneck_v1/Relu',
  'res3b1_branch2a':      'resnet_v1_101/block2/unit_2/bottleneck_v1/conv1/Relu',
  'res3b1_branch2b':      'resnet_v1_101/block2/unit_2/bottleneck_v1/conv2/Relu',
  'res3b1_branch2c':      'resnet_v1_101/block2/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res3b1':               'resnet_v1_101/block2/unit_2/bottleneck_v1/Relu',
  'res3b2_branch2a':      'resnet_v1_101/block2/unit_3/bottleneck_v1/conv1/Relu',
  'res3b2_branch2b':      'resnet_v1_101/block2/unit_3/bottleneck_v1/conv2/Relu',
  'res3b2_branch2c':      'resnet_v1_101/block2/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res3b2':               'resnet_v1_101/block2/unit_3/bottleneck_v1/Relu',
  'res3b3_branch2a':      'resnet_v1_101/block2/unit_4/bottleneck_v1/conv1/Relu',
  'res3b3_branch2b':      'resnet_v1_101/block2/unit_4/bottleneck_v1/conv2/Relu',
  'res3b3_branch2c':      'resnet_v1_101/block2/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res3b3':               'resnet_v1_101/block2/unit_4/bottleneck_v1/Relu',
  'res4a_branch1':        'resnet_v1_101/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm',
  'res4a_branch2a':       'resnet_v1_101/block3/unit_1/bottleneck_v1/conv1/Relu',
  'res4a_branch2b':       'resnet_v1_101/block3/unit_1/bottleneck_v1/conv2/Relu',
  'res4a_branch2c':       'resnet_v1_101/block3/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4a':                'resnet_v1_101/block3/unit_1/bottleneck_v1/Relu',
  'res4b1_branch2a':      'resnet_v1_101/block3/unit_2/bottleneck_v1/conv1/Relu',
  'res4b1_branch2b':      'resnet_v1_101/block3/unit_2/bottleneck_v1/conv2/Relu',
  'res4b1_branch2c':      'resnet_v1_101/block3/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b1':               'resnet_v1_101/block3/unit_2/bottleneck_v1/Relu',
  'res4b2_branch2a':      'resnet_v1_101/block3/unit_3/bottleneck_v1/conv1/Relu',
  'res4b2_branch2b':      'resnet_v1_101/block3/unit_3/bottleneck_v1/conv2/Relu',
  'res4b2_branch2c':      'resnet_v1_101/block3/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b2':               'resnet_v1_101/block3/unit_3/bottleneck_v1/Relu',
  'res4b3_branch2a':      'resnet_v1_101/block3/unit_4/bottleneck_v1/conv1/Relu',
  'res4b3_branch2b':      'resnet_v1_101/block3/unit_4/bottleneck_v1/conv2/Relu',
  'res4b3_branch2c':      'resnet_v1_101/block3/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b3':               'resnet_v1_101/block3/unit_4/bottleneck_v1/Relu',
  'res4b4_branch2a':      'resnet_v1_101/block3/unit_5/bottleneck_v1/conv1/Relu',
  'res4b4_branch2b':      'resnet_v1_101/block3/unit_5/bottleneck_v1/conv2/Relu',
  'res4b4_branch2c':      'resnet_v1_101/block3/unit_5/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b4':               'resnet_v1_101/block3/unit_5/bottleneck_v1/Relu',
  'res4b5_branch2a':      'resnet_v1_101/block3/unit_6/bottleneck_v1/conv1/Relu',
  'res4b5_branch2b':      'resnet_v1_101/block3/unit_6/bottleneck_v1/conv2/Relu',
  'res4b5_branch2c':      'resnet_v1_101/block3/unit_6/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b5':               'resnet_v1_101/block3/unit_6/bottleneck_v1/Relu',
  'res4b6_branch2a':      'resnet_v1_101/block3/unit_7/bottleneck_v1/conv1/Relu',
  'res4b6_branch2b':      'resnet_v1_101/block3/unit_7/bottleneck_v1/conv2/Relu',
  'res4b6_branch2c':      'resnet_v1_101/block3/unit_7/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b6':               'resnet_v1_101/block3/unit_7/bottleneck_v1/Relu',
  'res4b7_branch2a':      'resnet_v1_101/block3/unit_8/bottleneck_v1/conv1/Relu',
  'res4b7_branch2b':      'resnet_v1_101/block3/unit_8/bottleneck_v1/conv2/Relu',
  'res4b7_branch2c':      'resnet_v1_101/block3/unit_8/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b7':               'resnet_v1_101/block3/unit_8/bottleneck_v1/Relu',
  'res4b8_branch2a':      'resnet_v1_101/block3/unit_9/bottleneck_v1/conv1/Relu',
  'res4b8_branch2b':      'resnet_v1_101/block3/unit_9/bottleneck_v1/conv2/Relu',
  'res4b8_branch2c':      'resnet_v1_101/block3/unit_9/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b8':               'resnet_v1_101/block3/unit_9/bottleneck_v1/Relu',
  'res4b9_branch2a':      'resnet_v1_101/block3/unit_10/bottleneck_v1/conv1/Relu',
  'res4b9_branch2b':      'resnet_v1_101/block3/unit_10/bottleneck_v1/conv2/Relu',
  'res4b9_branch2c':      'resnet_v1_101/block3/unit_10/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b9':               'resnet_v1_101/block3/unit_10/bottleneck_v1/Relu',
  'res4b10_branch2a':     'resnet_v1_101/block3/unit_11/bottleneck_v1/conv1/Relu',
  'res4b10_branch2b':     'resnet_v1_101/block3/unit_11/bottleneck_v1/conv2/Relu',
  'res4b10_branch2c':     'resnet_v1_101/block3/unit_11/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b10':              'resnet_v1_101/block3/unit_11/bottleneck_v1/Relu',
  'res4b11_branch2a':     'resnet_v1_101/block3/unit_12/bottleneck_v1/conv1/Relu',
  'res4b11_branch2b':     'resnet_v1_101/block3/unit_12/bottleneck_v1/conv2/Relu',
  'res4b11_branch2c':     'resnet_v1_101/block3/unit_12/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b11':              'resnet_v1_101/block3/unit_12/bottleneck_v1/Relu',
  'res4b12_branch2a':     'resnet_v1_101/block3/unit_13/bottleneck_v1/conv1/Relu',
  'res4b12_branch2b':     'resnet_v1_101/block3/unit_13/bottleneck_v1/conv2/Relu',
  'res4b12_branch2c':     'resnet_v1_101/block3/unit_13/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b12':              'resnet_v1_101/block3/unit_13/bottleneck_v1/Relu',
  'res4b13_branch2a':     'resnet_v1_101/block3/unit_14/bottleneck_v1/conv1/Relu',
  'res4b13_branch2b':     'resnet_v1_101/block3/unit_14/bottleneck_v1/conv2/Relu',
  'res4b13_branch2c':     'resnet_v1_101/block3/unit_14/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b13':              'resnet_v1_101/block3/unit_14/bottleneck_v1/Relu',
  'res4b14_branch2a':     'resnet_v1_101/block3/unit_15/bottleneck_v1/conv1/Relu',
  'res4b14_branch2b':     'resnet_v1_101/block3/unit_15/bottleneck_v1/conv2/Relu',
  'res4b14_branch2c':     'resnet_v1_101/block3/unit_15/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b14':              'resnet_v1_101/block3/unit_15/bottleneck_v1/Relu',
  'res4b15_branch2a':     'resnet_v1_101/block3/unit_16/bottleneck_v1/conv1/Relu',
  'res4b15_branch2b':     'resnet_v1_101/block3/unit_16/bottleneck_v1/conv2/Relu',
  'res4b15_branch2c':     'resnet_v1_101/block3/unit_16/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b15':              'resnet_v1_101/block3/unit_16/bottleneck_v1/Relu',
  'res4b16_branch2a':     'resnet_v1_101/block3/unit_17/bottleneck_v1/conv1/Relu',
  'res4b16_branch2b':     'resnet_v1_101/block3/unit_17/bottleneck_v1/conv2/Relu',
  'res4b16_branch2c':     'resnet_v1_101/block3/unit_17/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b16':              'resnet_v1_101/block3/unit_17/bottleneck_v1/Relu',
  'res4b17_branch2a':     'resnet_v1_101/block3/unit_18/bottleneck_v1/conv1/Relu',
  'res4b17_branch2b':     'resnet_v1_101/block3/unit_18/bottleneck_v1/conv2/Relu',
  'res4b17_branch2c':     'resnet_v1_101/block3/unit_18/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b17':              'resnet_v1_101/block3/unit_18/bottleneck_v1/Relu',
  'res4b18_branch2a':     'resnet_v1_101/block3/unit_19/bottleneck_v1/conv1/Relu',
  'res4b18_branch2b':     'resnet_v1_101/block3/unit_19/bottleneck_v1/conv2/Relu',
  'res4b18_branch2c':     'resnet_v1_101/block3/unit_19/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b18':              'resnet_v1_101/block3/unit_19/bottleneck_v1/Relu',
  'res4b19_branch2a':     'resnet_v1_101/block3/unit_20/bottleneck_v1/conv1/Relu',
  'res4b19_branch2b':     'resnet_v1_101/block3/unit_20/bottleneck_v1/conv2/Relu',
  'res4b19_branch2c':     'resnet_v1_101/block3/unit_20/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b19':              'resnet_v1_101/block3/unit_20/bottleneck_v1/Relu',
  'res4b20_branch2a':     'resnet_v1_101/block3/unit_21/bottleneck_v1/conv1/Relu',
  'res4b20_branch2b':     'resnet_v1_101/block3/unit_21/bottleneck_v1/conv2/Relu',
  'res4b20_branch2c':     'resnet_v1_101/block3/unit_21/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b20':              'resnet_v1_101/block3/unit_21/bottleneck_v1/Relu',
  'res4b21_branch2a':     'resnet_v1_101/block3/unit_22/bottleneck_v1/conv1/Relu',
  'res4b21_branch2b':     'resnet_v1_101/block3/unit_22/bottleneck_v1/conv2/Relu',
  'res4b21_branch2c':     'resnet_v1_101/block3/unit_22/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res4b21':              'resnet_v1_101/block3/unit_22/bottleneck_v1/Relu',
  'res4b22_branch2a':     'resnet_v1_101/block3/unit_23/bottleneck_v1/conv1/Relu',
  'res4b22_branch2b':     'resnet_v1_101/block3/unit_23/bottleneck_v1/conv2/Relu',
  'res4b22_branch2c':     'resnet_v1_101/block3/unit_23/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',\
  'res4b22':              'resnet_v1_101/block3/unit_23/bottleneck_v1/Relu',
    
  'res5a_branch1':        'resnet_v1_101/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm',
    
  'res5a_branch2a':       'resnet_v1_101/block4/unit_1/bottleneck_v1/conv1/Relu',
  'res5a_branch2b':       'resnet_v1_101/block4/unit_1/bottleneck_v1/conv2/Relu',
  'res5a_branch2c':       'resnet_v1_101/block4/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res5a':                'resnet_v1_101/block4/unit_1/bottleneck_v1/Relu',
  'res5b_branch2a':       'resnet_v1_101/block4/unit_2/bottleneck_v1/conv1/Relu',
  'res5b_branch2b':       'resnet_v1_101/block4/unit_2/bottleneck_v1/conv2/Relu',
  'res5b_branch2c':       'resnet_v1_101/block4/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res5b':                'resnet_v1_101/block4/unit_2/bottleneck_v1/Relu',
  'res5c_branch2a':       'resnet_v1_101/block4/unit_3/bottleneck_v1/conv1/Relu',
  'res5c_branch2b':       'resnet_v1_101/block4/unit_3/bottleneck_v1/conv2/Relu',
  'res5c_branch2c':       'resnet_v1_101/block4/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm',
  'res5c':                'resnet_v1_101/block4/unit_3/bottleneck_v1/Relu',
  'pool5':                'resnet_v1_101/pool5',
  'fc1000':               'resnet_v1_101/logits/BiasAdd',
}

resnet_v1_101_tf_to_caffe = {}
for key, val in resnet_v1_101_caffe_to_tf.items():
    resnet_v1_101_tf_to_caffe[val] = key
    


class Net:
    def __init__(self, latest_checkpoint,
                 input_tensor_name=None,
                 output_tensor_name=None):
        """Helper to hold params and allow func like the original.

        :param graph: Location of the meta graph.
        :param weights: Location of the weights for the graph.
        :param input_tensor_name: The full tensor name (Include the dev
            placement and scope).
        :param output_tensor_name: The full tensor name (Include the dev
            placement and scope).
        """
        batch_size = 1
        num_class = 1000
        # Create tensorflow graph for evaluation
        eval_graph = tf.Graph()
        with eval_graph.as_default():
            images = tf.placeholder("float", [batch_size, 224, 224, 3], name="images")
            labels = tf.placeholder(tf.float32, [batch_size, num_class], name="labels")

            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                with slim.arg_scope([slim.batch_norm], is_training=False):
                    # is_training=False means batch-norm is not in training mode. Fixing batch norm layer.
                    # net is logit for resnet_v1. See is_training messing up issue: https://github.com/tensorflow/tensorflow/issues/4887
                    net, end_points = resnet_v1.resnet_v1_101(images, num_class, is_training=False)
                    print("net:", net)
            prob = tf.reshape(end_points['predictions'], (batch_size, num_class)) # after softmax

            init = tf.global_variables_initializer()

            ## Optimistic restore.
            reader = tf.train.NewCheckpointReader(latest_checkpoint)
            saved_shapes = reader.get_variable_to_shape_map()
            variables_to_restore = tf.global_variables()
            for var in variables_to_restore:
                if not var.name.split(':')[0] in saved_shapes:
                    print("WARNING. Saved weight not exists in checkpoint. Init var:", var.name)
                else:
                    # print("Load saved weight:", var.name)
                    pass

            var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables_to_restore
                                        if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    try:
                        curr_var = tf.get_variable(saved_var_name)
                        var_shape = curr_var.get_shape().as_list()
                        if var_shape == saved_shapes[saved_var_name]:
                            # print("restore var:", saved_var_name)
                            restore_vars.append(curr_var)
                    except ValueError:
                            print("Ignore due to ValueError on getting var:", saved_var_name) 
            saver = tf.train.Saver(restore_vars)
            
        self.sess = tf.Session(graph=eval_graph)
        self.sess.run(init)
        saver.restore(self.sess, latest_checkpoint)
        
        # with eval_graph.as_default():
        #     for i in tf.get_default_graph().get_operations():
        #         print(i.name)

        
        self.placeholder = images            
        self.softmax = prob
        
        with eval_graph.as_default():            
            get_tensor = tf.get_default_graph().get_tensor_by_name
            # Save trainables into params
            trainable_params = tf.trainable_variables()            
            layers = {}
            params = {}

            def add_to_layer(layer_name, activation_name):
                try:        
                    layers[layer_name] = get_tensor("{}:0".format(activation_name))                    
                except KeyError:
                    print("Activation Not Found.")
                    pass

        
            # for v in trainable_params:
                # print(v.name)
            #     if 'weights:0' == v.name[-9:]:
            #         print("add weights layer:", v.name\
             #              .replace("resnet_v1_101/", "")\
              #             .replace("bottleneck_v1/", ""))
               #      name = v.name.split('/')[0]
                #     params[name] = v
                 #    add_to_layer(name)

            # Pooling layers usually don't have a nice way of gathering.

            for n in tf.get_default_graph().as_graph_def().node:
                if n.name in resnet_v1_101_tf_to_caffe:                    
                    layer_name = resnet_v1_101_tf_to_caffe[n.name]
                    print("add caffe resnet layer ({}): {}".format(layer_name, n.name))
                    name = n.name
                    if "conv1/Relu" in name:
                        param_name = name.replace("Relu", "weights")
                    elif "bottleneck_v1/conv2/Relu" in name or "bottleneck_v1/conv1/Relu" in name:
                        param_name = name.replace("Relu", "weights")
                    elif "bottleneck_v1/conv3/BatchNorm/FusedBatchNorm" in name:
                        param_name = name.replace("bottleneck_v1/conv3/BatchNorm/FusedBatchNorm", "bottleneck_v1/conv3/weights")
                    elif "bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm" in name:
                        param_name = name.replace("bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm", "bottleneck_v1/shortcut/weights")
                    else:
                        param_name = name

                    param_tensor = get_tensor("{}:0".format(param_name))
                    print("param_name:", param_name)
                    print("param_tensor:", param_tensor)
                    params[layer_name] = param_tensor
                    add_to_layer(layer_name, name)
                else:
                    if "save" in n.name or "BatchNorm" in n.name or "Initializer" in n.name or "Regularizer" in n.name:
                        continue
                    print("skip ", n.name)

        # Get trainable params - 1 holds locations the other is a dummy script
        self.params = {}
        self._params = params
        self.layers = layers
        # Save empty dict into blobs
        self.blobs = {}

    def predict(self, img, oversample=False):
        
        img = _oversample(img, (224, 224))[4]
        # print("img after getting centcrop img:", img)
        # pos_img = (img - np.min(img))
        # norm = pos_img / np.max(pos_img)
        # print("normalized image visualization")
        # plt.imshow(norm)
        # plt.show()
        
        batch_img1 = img.reshape((1, 224, 224, 3))
        batch_images = (batch_img1,)
        if len(batch_images) == 1:
            batch_img = img.reshape((1, 224, 224, 3))
        else:
            batch_img = np.concatenate(batch_images, 0)
            
        
        batch_size = 1

        predictions = self.sess.run([self.softmax],
                                    feed_dict={self.placeholder: batch_img})[0]
        # shape of (batch_size, prob_per_class)
        # print("predictions shape:", predictions.shape)
        # print("predictions:", predictions)
        # for i in range(1000):
        #     if (predictions[0][i] > 0.005):
        #         print("idx: {}, score: {:.4f}".format(i, predictions[0][i]))
        
        # print("argmax prediction:", np.argmax(predictions))
        # print("max prediction:", np.max(predictions))

        if oversample:
            predictions = predictions.reshape((
                len(predictions) // NUM_OVER_SAMPLES,
                NUM_OVER_SAMPLES,
                -1
            )) # (10 * 1000 / 10, 10, 1)
            
            
            # Currently only meant to work with 1 image, luckily we have only 1
            # image. Transpose to match Caffe.
            predictions = predictions[0].T

        self.blobs['prob'] = Data(predictions) # shape of (num_batch, prob_per_class, 1)

        # Gather vars
        layers, params = self.sess.run([self.layers, self._params],
                                       feed_dict={self.placeholder: batch_img})

        # Run to update current activations and weights
        for k, v in layers.items():
            self.blobs[k] = Data(_permute_blobs_to_caffe(v))

        for k, v in params.items():
            self.params[k] = [Data(_permute_params_to_caffe(v))]


class Data:

    def __init__(self, d):
        self.data = d
