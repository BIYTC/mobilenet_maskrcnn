import torch.nn as nn
from maskrcnn_benchmark.modeling.make_layers import inverted_residual_sequence, conv2d_bn_relu6


class MobileNetV2(nn.Module):
    def __init__(self, args):
        super(MobileNetV2, self).__init__()

        # s1, s2 = 2, 2
        # if args.DOWNSAMPLING == 16:
        #     s1, s2 = 2, 1
        # elif args.DOWNSAMPLING == 8:
        #     s1, s2 = 1, 1

        # Network is created here, then will be unpacked into nn.sequential
        self.network_settings = [{'t': -1, 'c': 32, 'n': 1, 's': 1},
                                 {'t': 1, 'c': 16, 'n': 1, 's': 1},
                                 {'t': 6, 'c': 24, 'n': 2, 's': 1},
                                 {'t': 6, 'c': 32, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 64, 'n': 4, 's': 2},
                                 {'t': 6, 'c': 128, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 256, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 512, 'n': 1, 's': 2},
                                 {'t': None, 'c': 1280, 'n': 1, 's': 1}]
        self.num_classes = args.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1

        ###############################################################################################################

        # Feature Extraction part
        # Layer 0
        self.network = [
            conv2d_bn_relu6(args.NUM_CHANNELS,
                            int(self.network_settings[0]['c'] * args.WIDTH_MULTIPLIER),
                            args.KERNEL_SIZE,
                            self.network_settings[0]['s'], args.DROUPOUT_PROB)]

        # Layers from 1 to 7
        for i in range(1, 8):
            self.network.extend(
                inverted_residual_sequence(
                    int(self.network_settings[i - 1]['c'] * args.WIDTH_MULTIPLIER),
                    int(self.network_settings[i]['c'] * args.WIDTH_MULTIPLIER),
                    self.network_settings[i]['n'], self.network_settings[i]['t'],
                    args.KERNEL_SIZE, self.network_settings[i]['s']))

        # Last layer before flattening
        # self.network.append(
        #     conv2d_bn_relu6(int(self.network_settings[7]['c'] * args.WIDTH_MULTIPLIER),
        #                     int(self.network_settings[8]['c'] * args.WIDTH_MULTIPLIER), 1,
        #                     self.network_settings[8]['s'],
        #                     args.DROUPOUT_PROB))

        ###############################################################################################################

        # Classification part
        # self.network.append(nn.Dropout2d(args.DROUPOUT_PROB, inplace=True))
        # self.network.append(nn.AvgPool2d(
        #     (args.IMG_HEIGHT // args.DOWNSAMPLING, args.IMG_WIDTH // args.DOWNSAMPLING)))
        # self.network.append(nn.Dropout2d(args.DROUPOUT_PROB, inplace=True))
        # self.network.append(
        #     nn.Conv2d(int(self.network_settings[8]['c'] * args.WIDTH_MULTIPLIER), self.num_classes,
        #               1, bias=True))

        self.network = nn.Sequential(*self.network)

        self.initialize()

    # def forward(self, x):
    #     # Debugging mode
    #     # for op in self.network:
    #     #     x = op(x)
    #     #     print(x.shape)
    #     x = self.network(x)
    #     x = x.view(-1, self.num_classes)

    #    return x

    def forward(self, x):
        outputs = []
        # x = self.network(x)
        for idx, stage_name in enumerate(self.network):
            x = stage_name(x)
            if idx in [10, 13, 16, 17]:
                outputs.append(x)
        return outputs

    def initialize(self):
        """Initializes the model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
