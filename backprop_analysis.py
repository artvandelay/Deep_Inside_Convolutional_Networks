import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import matplotlib.cm as cm
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


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data,cmap=cm.gray)



#Perform a forward pass with the data as the input image
prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically

#Perform a backward pass for the cat class (281)
bw = net.backward(**{net.outputs[0]: caffeLabel})
diff = bw['data']


# Plot each derivative of each layer and save each fig.
feat = net.blobs['conv1'].diff[0]
vis_square(feat, padval=1)
plt.title('conv1')
plt.savefig('ps3part3_conv1.png')

feat = net.blobs['conv2'].diff[0]
vis_square(feat, padval=1)
plt.title('conv2')
plt.savefig('ps3part3_conv2.png')

feat = net.blobs['conv3'].diff[0]
vis_square(feat, padval=1)
plt.title('conv3')
plt.savefig('ps3part3_conv3.png')

feat = net.blobs['conv4'].diff[0]
vis_square(feat, padval=1)
plt.title('conv4')
plt.savefig('ps3part3_conv4.png')

feat = net.blobs['conv5'].diff[0]
vis_square(feat, padval=1)
plt.title('conv5')
plt.savefig('ps3part3_conv5.png')

feat = net.blobs['pool1'].diff[0]
vis_square(feat, padval=1)
plt.title('pool1')
plt.savefig('ps3part3_pool1.png')

feat = net.blobs['pool2'].diff[0]
vis_square(feat, padval=1)
plt.title('pool2')
plt.savefig('ps3part3_pool2.png')

feat = net.blobs['pool5'].diff[0]
vis_square(feat, padval=1)
plt.title('pool5')
plt.savefig('ps3part3_pool5.png')

feat = net.blobs['norm1'].diff[0]
vis_square(feat, padval=1)
plt.title('norm1')
plt.savefig('ps3part3_norm1.png')

feat = net.blobs['norm2'].diff[0]
vis_square(feat, padval=1)
plt.title('norm2')
plt.savefig('ps3part3_norm2.png')

feat = net.blobs['fc6'].diff[0]
vis_square(feat, padval=1)
plt.title('fc6')
plt.savefig('ps3part3_fc6.png')

feat = net.blobs['fc7'].diff[0]
vis_square(feat, padval=1)
plt.title('fc7')
plt.savefig('ps3part3_fc7.png')


feat = net.blobs['fc8'].diff[0]
vis_square(feat, padval=1)
plt.title('fc8')
plt.savefig('ps3part3_fc8.png')

