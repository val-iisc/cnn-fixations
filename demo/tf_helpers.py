"""
    Helper functions to hide Tensorflow as Caffe.
"""
import numpy as np
import tensorflow as tf

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
        data = np.transpose(data, (3, 2, 0, 1))
    elif data.ndim == 2:
        data = np.transpose(data)
    else:
        raise(ValueError, 'Well this is unexpected...')

    return data


def _permute_blobs_to_caffe(data):
    if data.ndim == 4:
        if data.ndim == 4:
            data = np.transpose(data, (0, 3, 1, 2))
        elif data.ndim == 2:
            data = np.transpose(data)
        else:
            raise (ValueError, 'Well this is unexpected...')

    return data


class Net:
    def __init__(self, graph, weights,
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
        self.sess = tf.Session()
        new_saver = tf.train.import_meta_graph(graph)
        new_saver.restore(self.sess, weights)

        get_tensor = tf.get_default_graph().get_tensor_by_name
        # Get the initial place holder, else default
        if input_tensor_name:
            self.placeholder = get_tensor(input_tensor_name)
        else:
            self.placeholder = get_tensor('Placeholder:0')

        if output_tensor_name:
            self.softmax = get_tensor(output_tensor_name)
        else:
            self.softmax = get_tensor('Softmax:0')

        # Save trainables into params
        trainable_params = tf.trainable_variables()
        layers = {}
        params = {}

        def add_to_layer(name):
            try:
                layers[name] = get_tensor("{}:0".format(name))
            except KeyError:
                try:
                    layers[name] = get_tensor("{}/Relu:0".format(name))
                except KeyError:
                    print("Activation Not Found.")
                    pass

        for v in trainable_params:
            if 'weight' in v.name:
                print("add weight layer:", v.name)
                name = v.name.split('/')[0]
                params[name] = v
                add_to_layer(name)

        # Pooling layers usually don't have a nice way of gathering.
        for n in tf.get_default_graph().as_graph_def().node:
            if 'pool' in n.name:
                print("add pool layer:", n.name)
                v = get_tensor("{}:0".format(n.name))
                name = n.name.split('/')[0]
                params[name] = v
                add_to_layer(name)

        # Get trainable params - 1 holds locations the other is a dummy script
        self.params = {}
        self._params = params
        self.layers = layers
        # Save empty dict into blobs
        self.blobs = {}

    def predict(self, img, oversample=True):
        imgs = _oversample(img, (224, 224))
        print("imgs:",  imgs)
        print("imgs shape:",  imgs.shape)

        predictions = self.sess.run([self.softmax],
                                    feed_dict={self.placeholder: imgs})[0]

        if oversample:
            predictions = predictions.reshape((len(predictions) //
                                               NUM_OVER_SAMPLES,
                                               NUM_OVER_SAMPLES, -1))
        # Currently only meant to work with 1 image, luckily we have only 1
        # image. Transpose to match Caffe.
        self.blobs['prob'] = Data(predictions[0].T)

        # Gather vars
        
        print("imgs:",  imgs)
        print("imgs shape:",  imgs.shape)
        layers, params = self.sess.run([self.layers, self._params],
                                       feed_dict={self.placeholder: imgs})

        # Run to update current activations and weights
        for k, v in layers.items():
            self.blobs[k] = Data(_permute_blobs_to_caffe(v))

        for k, v in params.items():
            self.params[k] = [Data(_permute_params_to_caffe(v))]


class Data:

    def __init__(self, d):
        self.data = d
