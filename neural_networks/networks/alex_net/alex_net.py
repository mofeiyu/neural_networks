import tensorflow as tf

class AlexNet():
    def __init__(self, class_num):
        self._class_num = class_num
        self._parameters = {}
        
    def _initialize_parameters(self, X_channel):
        conv_W1 = tf.get_variable("conv_W1", [11, 11, X_channel, 96])
        conv_b1 = tf.get_variable("conv_b1", [96])
        conv_W2 = tf.get_variable("conv_W2", [5, 5, 96, 256])
        conv_b2 = tf.get_variable("conv_b2", [256])
        conv_W3 = tf.get_variable("conv_W3", [3, 3, 256, 384])
        conv_b3 = tf.get_variable("conv_b3", [384])
        conv_W4 = tf.get_variable("conv_W4", [3, 3, 384, 384])
        conv_b4 = tf.get_variable("conv_b4", [384])
        conv_W5 = tf.get_variable("conv_W5", [3, 3, 384, 256])
        conv_b5 = tf.get_variable("conv_b5", [256])         
        self._parameters = {"conv_W1": conv_W1, "conv_b1": conv_b1,
                      "conv_W2": conv_W2, "conv_b2": conv_b2,
                      "conv_W3": conv_W3, "conv_b3": conv_b3,
                      "conv_W4": conv_W4, "conv_b4": conv_b4,
                      "conv_W5": conv_W5, "conv_b5": conv_b5,}
        
    def forward(self, X):
        X_channel = X.shape[3]
        self._initialize_parameters(X_channel)                      
        conv_W1 = self._parameters["conv_W1"]
        conv_b1 = self._parameters["conv_b1"]
        conv_W2 = self._parameters["conv_W2"]
        conv_b2 = self._parameters["conv_b2"]
        conv_W3 = self._parameters["conv_W3"]
        conv_b3 = self._parameters["conv_b3"]
        conv_W4 = self._parameters["conv_W4"]
        conv_b4 = self._parameters["conv_b4"]
        conv_W5 = self._parameters["conv_W5"]
        conv_b5 = self._parameters["conv_b5"]
        
        conv1 = tf.nn.bias_add(tf.nn.conv2d(X, conv_W1, strides=[1, 4, 4, 1], padding = "VALID"), conv_b1)
        relu1 = tf.nn.relu(conv1, "relu1")
        norm1 = tf.nn.local_response_normalizationl(relu1,depth_radius = 2, alpha =2e-05, beta = 0.75, bias = 1.0, name = "norm1")
        pool1 = tf.nn.max_pool(norm1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "VALID", name = "pool1")
        
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, conv_W2, strides=[1, 1, 1, 1], padding = "same"), conv_b2)
        relu2 = tf.nn.relu(conv2, "relu1")
        norm2 = tf.nn.local_response_normalizationl(relu2, depth_radius = 2, alpha =2e-05, beta = 0.75, bias = 1.0, name = "norm2")
        pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "VALID", name = "pool2")
        
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, conv_W3, strides=[1, 1, 1, 1], padding = "same"), conv_b3)
        relu3 = tf.nn.relu(conv3, "relu3")

        conv4 = tf.nn.bias_add(tf.nn.conv2d(relu3, conv_W4, strides=[1, 1, 1, 1], padding = "same"), conv_b4)
        relu4 = tf.nn.relu(conv4, "relu4")

        conv5 = tf.nn.bias_add(tf.nn.conv2d(relu4, conv_W5, strides=[1, 1, 1, 1], padding = "same"), conv_b5)
        relu5 = tf.nn.relu(conv5, "relu5") 
        pool5 = tf.nn.max_pool(relu5, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "VALID", name = "pool5")
             
        pool5 = tf.contrib.layers.flatten(pool5)
        fc6 = tf.contrib.layers.fully_connected(pool5, 4096)
        fc7 = tf.contrib.layers.fully_connected(fc6, 4096)
        fc8 = tf.contrib.layers.fully_connected(fc7, self._class_num, activation_fn=None)
                
        return fc8