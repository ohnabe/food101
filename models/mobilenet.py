import chainer
import chainer.functions as F
import chainer.links as L
import cupy

class ConvBN(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvBN, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(inp, oup, 3, stride=stride, pad=1, nobias=True)
            self.bn = L.BatchNormalization(oup)

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        return h


class ConvDW(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvDW, self).__init__()
        with self.init_scope():
            self.conv_dw = L.DepthwiseConvolution2D(inp, 1, 3, stride=stride, pad=1, nobias=True)
            self.bn_dw = L.BatchNormalization(inp)
            self.conv_sep = L.Convolution2D(inp, oup, 1, stride=1, pad=0, nobias=True)
            self.bn_sep = L.BatchNormalization(oup)

    def __call__(self, x):
        h = F.relu(self.bn_dw(self.conv_dw(x)))
        h = F.relu(self.bn_sep(self.conv_sep(h)))
        return h


# Network definition
class MobileNet(chainer.Chain):
    def __init__(self, n_class):
        super(MobileNet, self).__init__()
        with self.init_scope():
            self.conv_bn = ConvBN(3, 32, 2)
            self.conv_ds_2 = ConvDW(32, 64, 1)
            self.conv_ds_3 = ConvDW(64, 128, 2)
            self.conv_ds_4 = ConvDW(128, 128, 1)
            self.conv_ds_5 = ConvDW(128, 256, 2)
            self.conv_ds_6 = ConvDW(256, 256, 1)
            self.conv_ds_7 = ConvDW(256, 512, 2)

            self.conv_ds_8 = ConvDW(512, 512, 1)
            self.conv_ds_9 = ConvDW(512, 512, 1)
            self.conv_ds_10 = ConvDW(512, 512, 1)
            self.conv_ds_11 = ConvDW(512, 512, 1)
            self.conv_ds_12 = ConvDW(512, 512, 1)

            self.conv_ds_13 = ConvDW(512, 1024, 2)
            self.conv_ds_14 = ConvDW(1024, 1024, 1)

            self.fc = L.Linear(None, n_class)

    def __call__(self, x, t):
        h = self.forward(x)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'accuracy': accuracy
        }, self)
        return loss


    def forward(self, x):
        h = self.conv_bn(x)
        h = self.conv_ds_2(h)
        h = self.conv_ds_3(h)
        h = self.conv_ds_4(h)
        h = self.conv_ds_5(h)
        h = self.conv_ds_6(h)
        h = self.conv_ds_7(h)
        h = self.conv_ds_8(h)
        h = self.conv_ds_9(h)
        h = self.conv_ds_10(h)
        h = self.conv_ds_11(h)
        h = self.conv_ds_12(h)
        h = self.conv_ds_13(h)
        h = self.conv_ds_14(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        #h = F.softmax(h)
        return h

