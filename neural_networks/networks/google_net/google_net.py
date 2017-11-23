import tensorflow as tf

class GoogleNetV1():
    def __init__(self, class_num):
        self._class_num = class_num
        self._parameters = {}
        
    def _get_variable_block_1(self, layer_num, f, n_p, n_c):
        W = tf.get_variable("W" + str(layer_num), [f, f, n_p, n_c])
        b = tf.get_variable("b" + str(layer_num), [n_c])
        self._parameters = {"W" + str(layer_num): W, "b" +str(layer_num): b}
        
    def _get_variable_block_2(self, layer_num, n_p, n_c):
        # n_c = [#1x1, #3X3reduce, #3x3, #5x5reduce, #5x5, #pool_proj]
        # 1x1 conv 
        W_1 = tf.get_variable("W" + str(layer_num) + '_1', [1, 1, n_p, n_c[0]])
        b_1 = tf.get_variable("b"+ str(layer_num) + '_1', n_c[0])
        self._parameters = {"W" + str(layer_num) + '_1': W_1, "b" +str(layer_num)+ '_1': b_1}
        # 1x1 conv before 3x3 conv, (3X3reduce)
        W_3_1 = tf.get_variable("W" + str(layer_num) + '_3_1', [1, 1, n_p, n_c[1]])
        b_3_1 = tf.get_variable("b"+ str(layer_num) + '_3_1', n_c[1])
        self._parameters = {"W" + str(layer_num) + '_3_1': W_3_1, "b" +str(layer_num)+ '_3_1': b_3_1}
        # 1x1 conv before 5x5 conv, (5X5reduce)
        W_5_1 = tf.get_variable("W" + str(layer_num) + '_5_1', [1, 1, n_p, n_c[3]])
        b_5_1 = tf.get_variable("b"+ str(layer_num) + '_5_1', n_c[3])
        self._parameters = {"W" + str(layer_num) + '_5_1': W_5_1, "b" +str(layer_num)+ '_5_1': b_5_1}
        # 3x3 conv       
        W_3 = tf.get_variable("W" + str(layer_num + 1) + '_3', [3, 3, n_c[1], n_c[2]])
        b_3 = tf.get_variable("b"+ str(layer_num + 1) + '_3', n_c[2])
        self._parameters = {"W" + str(layer_num + 1) + '_3': W_3, "b" +str(layer_num +1)+ '_3': b_3}
        # 5x5 conv
        W_5 = tf.get_variable("W" + str(layer_num + 1) + '_5', [5, 5, n_c[3], n_c[4]])
        b_5 = tf.get_variable("b"+ str(layer_num + 1) + '_5', n_c[4])
        self._parameters = {"W" + str(layer_num + 1) + '_5': W_5, "b" +str(layer_num + 1)+ '_5': b_5}
        # 1x1 conv after maxpool , (pool_proj)
        W_p_1 = tf.get_variable("W" + str(layer_num + 1) + '_p_1', [5, 5, n_p, n_c[5]])
        b_p_1 = tf.get_variable("b"+ str(layer_num + 1) + '_p_1', n_c[5])
        self._parameters = {"W" + str(layer_num + 1) + '_p_1': W_p_1, "b" + str(layer_num + 1)+ '_p_1': b_p_1}
            
    def _initialize_parameters(self, X_channel):
        # layer 1 - 3
        self._get_variable_block_1(1, 7, X_channel, 64)
        self._get_variable_block_1(2, 1, 64, 64)
        self._get_variable_block_1(3, 3, 64, 192)
        # stage 2 layer 4-21
        self._get_variable_block_2(4, 192, [64, 96, 128, 16, 32, 32])
        self._get_variable_block_2(6, 256, [128, 128, 192, 32, 96, 64])
        self._get_variable_block_2(8, 480, [192, 96, 208, 16, 48, 64])
        self._get_variable_block_2(10, 512, [160, 112, 224, 24, 64, 64])
        self._get_variable_block_2(12, 512, [128, 128, 256, 24, 64, 64])
        self._get_variable_block_2(14, 512, [112, 144, 288, 32, 64, 64])
        self._get_variable_block_2(16, 528, [256, 160, 320, 32, 128, 128])
        self._get_variable_block_2(18, 832, [256, 160, 320, 32, 128, 128])
        self._get_variable_block_2(20, 832, [384, 192, 384, 48, 128, 128])
        
        
    def forward(self, X):       
        X_channel = X.shape[3]
        self._initialize_parameters(self, X_channel)
        # stage 1 layer 1-3
        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W1"], strides=[1, 2, 2, 1], padding = "VALID"), self._parameters["b1"])
        X = tf.nn.relu(X)
        X = tf.nn.max_pool(X, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W2"], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b2"])
        X = tf.nn.relu(X)
        X = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W3"], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b3"])
        X = tf.nn.relu(X)                
        X = tf.nn.local_response_normalization(X,depth_radius = 3)
        X = tf.nn.max_pool(X, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
        # stage 2 layer 4-21
        X = self._inception_block(X, 4)
        X = self._inception_block(X, 6)
        X = tf.nn.max_pool(X, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
        X = self._inception_block(X, 8)
        X = self._inception_block(X, 10)
        X = self._inception_block(X, 12)
        X = self._inception_block(X, 14)
        X = self._inception_block(X, 16)
        X = tf.nn.max_pool(X, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
        X = self._inception_block(X, 18)
        X = self._inception_block(X, 20)
        
        # stage 3
        X = tf.nn.avg_pool(X, ksize = (1,7,7,1), strides = (1,1,1,1), padding = "same")
        X = tf.contrib.layers.flatten(X)
        X = tf.nn.dropout(X, keep_prob=0.4)
        X = tf.contrib.layers.fully_connected(X, self._class_num, activation_fn=None)
        return X
    
    def _inception_block(self, X, layer_num):
        X_1 = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W" + str(layer_num) + '_1'], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b" + str(layer_num) + '_1'])
        X_1 = tf.nn.relu(X_1)

        X_3_1 = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W" + str(layer_num) + '_3_1'], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b" + str(layer_num) + '_3_1'])
        X_3_1 = tf.nn.relu(X_3_1)
        X_3 = tf.nn.bias_add(tf.nn.conv2d(X_3_1, self._parameters["W" + str(layer_num+1) + '_3'], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b" + str(layer_num+1) + '_3'])
        X_3 = tf.nn.relu(X_3)

        X_5_1 = tf.nn.bias_add(tf.nn.conv2d(X, self._parameters["W" + str(layer_num) + '_5_1'], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b" + str(layer_num) + '_5_1'])
        X_5_1 = tf.nn.relu(X_5_1)
        X_5 = tf.nn.bias_add(tf.nn.conv2d(X_5_1, self._parameters["W" + str(layer_num+1) + '_5'], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b" + str(layer_num+1) + '_5'])
        X_5 = tf.nn.relu(X_5)

        X_p = tf.nn.max_pool(X, ksize = [1,3,3,1], strides = [1,1,1,1], padding = 'same')
        X_p_1 = tf.nn.bias_add(tf.nn.conv2d(X_p, self._parameters["W" + str(layer_num + 1) + '_p_1'], strides=[1, 1, 1, 1], padding = "same"), self._parameters["b" + str(layer_num + 1) + '_p_1'])
        X_p_1 = tf.nn.relu(X_p_1)
        
        X = tf.concat(3,[X_1, X_3, X_5, X_p])
        return X              