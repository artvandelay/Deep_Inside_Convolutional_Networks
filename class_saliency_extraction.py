import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
import matplotlib.cm as cm

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy_fc8.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'images/cat.jpg'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
input_image = caffe.io.load_image(IMAGE_FILE)
input_image = input_image

n_iterations = 10000
label_index = 281  # Index for cat class
caffe_data = np.random.random((1,3,227,227))
caffeLabel = np.zeros((1,1000,1,1))
caffeLabel[0,label_index,0,0] = 1;

def visSquare(data1, padsize=1, padval=0):
    data = copy.deepcopy(data1) 
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show(block=False)

    return data

#Perform a forward pass with the data as the input image
pred = net.predict([input_image])

#Perform a backward pass for the cat class (281)
bw = net.backward(**{net.outputs[0]: caffeLabel})
diff = bw['data']


# Find the saliency map as described in the paper. Normalize the map and assign it to variabe "saliency"
diff = np.abs(diff)
diff /= diff.max()
diff_sq = np.squeeze(diff)
saliency = np.amax(diff_sq,axis=0)

#display the saliency map
plt.subplot(1,2,1)
plt.imshow(saliency, cmap=cm.gray_r)
plt.subplot(1,2,2)
plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0]))
#plt.show()
plt.savefig('ps3part2.png')

