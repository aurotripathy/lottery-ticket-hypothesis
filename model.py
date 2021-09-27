
import torch.nn as nn
from types import MethodType


def create_maskable_module(module):

    def set_mask(self, mask):
        self.mask = mask.clone().detach()
        self.weight.data *= self.mask.data

    def forward_hook(forward_method):
        def forward(self, X):
            if hasattr(self, 'mask'):
                self.set_mask(self.mask)
            return forward_method(X)
        return forward

    module.set_mask = MethodType(set_mask, module)
    module.forward = MethodType(forward_hook(module.forward), module)
    return module


class Lenet_300_100(nn.Module):
    def __init__(self):
        super(Lenet_300_100, self).__init__()
        self.fc300 = create_maskable_module(nn.Linear(28 * 28, 300))
        self.relu300 = nn.ReLU(inplace=True)

        self.fc100 = create_maskable_module(nn.Linear(300, 100))
        self.relu100 = nn.ReLU(inplace=True)

        self.fc10 = create_maskable_module(nn.Linear(100, 10))

        self.inited_params = self.state_dict()

    def forward(self, x):
        x = x.view(x.size(0), 28 * 28)
        x = self.fc300(x)
        x = self.relu300(x)
        x = self.fc100(x)
        x = self.relu100(x)
        logits = self.fc10(x)
        return logits

    def set_masks(self, masks):
        self.fc300.set_mask(masks[0])
        self.fc100.set_mask(masks[1])
        self.fc10.set_mask(masks[2])

    def reset_parameters(self):
        self.load_state_dict(self.inited_params)

    def init_parameters(self):
        for name in self.init_params.keys():
            yield self.init_params[name]
