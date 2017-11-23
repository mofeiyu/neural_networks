import logging
from neural_networks.networks.alex_net.alex_net import AlexNet
from neural_networks.networks.res_net.res_net import ResNet
from neural_networks.networks.google_net.google_net import GoogleNet
from neural_networks.networks.vgg_net.vgg_net import VggNet16

class NeuralNetworkFactory( ):
    @staticmethod
    def get_neural_network(nn, n_y):
        if nn == "alex":
            return AlexNet(n_y)
        elif nn == "res":
            return ResNet34(n_y)
        elif nn == "google":
            return GoogleNetV1(n_y)
        else :
            logging("Sorry, without this neural network here!")