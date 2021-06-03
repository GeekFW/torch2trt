from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.clone')
@tensorrt_converter('torch.Tensor.clone')
def convert_clone(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    output._trt = input._trt  # Doesn't actually clone yet


class Clone(torch.nn.Module):
    def __init__(self, memory_format=torch.preserve_format):
        super(Clone, self).__init__()
        self.memory_format = memory_format

    def forward(self, x):
        return x.clone(memory_format=self.memory_format)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_clone_Clone():
    return Clone()


class TorchClone(torch.nn.Module):
    def __init__(self, memory_format=torch.preserve_format):
        super(TorchClone, self).__init__()
        self.memory_format = memory_format

    def forward(self, x):
        return torch.clone(x, memory_format=self.memory_format)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_clone_TorchClone():
    return TorchClone()
