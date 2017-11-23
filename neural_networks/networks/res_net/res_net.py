import tensorflow as tf

class ResNet34():
    def __init__(self, class_num):
        self._class_num = class_num
        self._parameters = {}
    
    def _initialize_parameters(self, X_channel):
        # stage 1
        W1 = tf.get_variable("W1", [7, 7, X_channel, 64])
        b1 = tf.get_variable("b1", [64])
        self._parameters = {"W1": W1, "b1":  b1}
        
        # stage 2
        for l in range(2,8):
            W = tf.get_variable("W"+ str(l), [3, 3, 64, 64])
            b = tf.get_variable("b"+ str(l), [64])
            self._parameters = {"W" + str(l): W,  "b" + str(l): b}
        
        # stage 3
        W = tf.get_variable("W8", [3, 3, 64, 128])
        b = tf.get_variable("b8", [128])
        self._parameters = {"W8" : W, "b8": b}               
        for l in range(9,16):
            W = tf.get_variable("W"+ str(l), [3, 3, 128, 128])
            b = tf.get_variable("b"+ str(l), [128])
            self._parameters = {"W" + str(l): W, "b" + str(l): b}
        # stage 4
        W = tf.get_variable("W16", [3, 3, 128, 256])
        b = tf.get_variable("b16", [256])
        self._parameters = {"W16" : W, "b16": b}
        for l in range(17,28):
            W = tf.get_variable("W"+ str(l), [3, 3, 256, 256])
            b = tf.get_variable("b"+ str(l), [256])
            self._parameters = {"W" + str(l): W, "b" + str(l): b}
        # stage 5
        W = tf.get_variable("W28", [3, 3, 256, 512])
        b = tf.get_variable("b28", [512])
        self._parameters = {"W28" : W, "b28": b}         
        for l in range(29,34):
            W = tf.get_variable("W"+ str(l), [3, 3, 512, 512])
            b = tf.get_variable("b"+ str(l), [512])
            self._parameters = {"W" + str(l): W, "b" + str(l): b}      
                                        
    def forward(self,X):
        
        X_channel = X.shape[3]
        self._initialize_parameters(self, X_channel)
        # stage 1
        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W1"], strides=[1, 2, 2, 1], padding = "VALID"), self._parameters["b1"])
        X = tf.nn.local_response_normalization(X,depth_radius = 3)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool(X, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'same')
        
        # stage 2 layer 2 - 8
        X = self.identity_block(X, 2)
        X = self.identity_block(X, 4)
        X = self.identity_block(X, 6)

        # stage 3 layer 9 - 15
        X = self.identity_block_2(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = self.identity_block(X, 10)
        X = self.identity_block(X, 12)
        X = self.identity_block(X, 14)
        
        # stage 4 layer 16 - 27
        X = self.identity_block_2(X, 16)
        X = self.identity_block(X, 18)
        X = self.identity_block(X, 20)
        X = self.identity_block(X, 22)
        X = self.identity_block(X, 24)
        X = self.identity_block(X, 26)
        
        # stage 5 layer 28 - 33
        X = self.identity_block_2(X, 28)
        X = self.identity_block(X, 30)
        X = self.identity_block(X, 32)
        
        # stage 6 layer 34
        X = tf.nn.avg_pool(X, ksize = (1,2,2,1), strides = (1,2,2,1), padding = "VALID")
        X = tf.contrib.layers.flatten(X)
        X = tf.contrib.layers.fully_connected(X, self._class_num, activation_fn=None)
        return X

    def identity_block(self, X, layer_start_num):
        l = layer_start_num
        X_shortcut = X

        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W"+ str(l)], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b"+ str(l)])
        X = tf.nn.local_response_normalization(X,depth_radius = 3)
        X = tf.nn.relu(X)

        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W"+ str(l+1)], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b"+ str(l+1)])
        X = tf.nn.local_response_normalization(X,depth_radius = 3)
   
        X = tf.add(X, X_shortcut)
        X = tf.nn.relu(X)        
        return X
    
    def identity_block_2(self, X, layer_start_num):
        l = layer_start_num
        X_shortcut = X

        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W"+ str(l)], strides=[1, 2, 2, 1], padding = "same"), self._parameters["b"+ str(l)])
        X = tf.nn.local_response_normalization(X,depth_radius = 3)
        X = tf.nn.relu(X)

        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W"+ str(l+1)], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b"+ str(l+1)])
        X = tf.nn.local_response_normalization(X,depth_radius = 3)
   
        X_shortcut_pad = tf.pad(X_shortcut, paddings = [[0,0],[0,0],[0,0],[0,X_shortcut.shape[3]]]) 
        X = tf.add(X, X_shortcut_pad)
        X = tf.nn.relu(X)
        return X