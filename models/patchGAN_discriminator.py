import torch.nn as nn
import numpy as np
from models.normlizations import get_norm_layer


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers_D=3, norm_D='instance', no_ganFeat_loss=True):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers_D (int)  -- the number of conv layers in the discriminator
            norm_D      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = ndf

        norm_layer = get_norm_layer(norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        self.no_ganFeat_loss = no_ganFeat_loss
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
