
from omegaconf import OmegaConf
from typing import Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


from math import log2, sqrt
from functools import partial
from typing import Optional, Union, Tuple, List
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import filter2d
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


enabled = True
weight_gradients_disabled = False


@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled

    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if could_use_op(input):
        return conv2d_gradfix(
            transpose=False,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups,
        ).apply(input, weight, bias)

    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if could_use_op(input):
        return conv2d_gradfix(
            transpose=True,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ).apply(input, weight, bias)

    return F.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def could_use_op(input):
    if (not enabled) or (not torch.backends.cudnn.enabled):
        return False

    if input.device.type != "cuda":
        return False

    if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8."]):
        return True

    warnings.warn(
        f"conv2d_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.conv2d()."
    )

    return False


def ensure_tuple(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim

    return xs


conv2d_gradfix_cache = dict()


def conv2d_gradfix(
    transpose, weight_shape, stride, padding, output_padding, dilation, groups
):
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = ensure_tuple(stride, ndim)
    padding = ensure_tuple(padding, ndim)
    output_padding = ensure_tuple(output_padding, ndim)
    dilation = ensure_tuple(dilation, ndim)

    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in conv2d_gradfix_cache:
        return conv2d_gradfix_cache[key]

    common_kwargs = dict(
        stride=stride, padding=padding, dilation=dilation, groups=groups
    )

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]

        return [
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    class Conv2d(autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            if not transpose:
                out = F.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)

            else:
                out = F.conv_transpose2d(
                    input=input,
                    weight=weight,
                    bias=bias,
                    output_padding=output_padding,
                    **common_kwargs,
                )

            ctx.save_for_backward(input, weight)

            return out

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = None, None, None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape
                )
                grad_input = conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                ).apply(grad_output, weight, None)

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))

            return grad_input, grad_weight, grad_bias

    class Conv2dGradWeight(autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            op = torch._C._jit_get_operation(
                "aten::cudnn_convolution_backward_weight"
                if not transpose
                else "aten::cudnn_convolution_transpose_backward_weight"
            )
            flags = [
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32,
            ]
            grad_weight = op(
                weight_shape,
                grad_output,
                input,
                padding,
                stride,
                dilation,
                groups,
                *flags,
            )
            ctx.save_for_backward(grad_output, input)

            return grad_weight

        @staticmethod
        def backward(ctx, grad_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_grad_output, grad_grad_input = None, None

            if ctx.needs_input_grad[0]:
                grad_grad_output = Conv2d.apply(input, grad_grad_weight, None)

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape
                )
                grad_grad_input = conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                ).apply(grad_output, grad_grad_weight, None)

            return grad_grad_output, grad_grad_input

    conv2d_gradfix_cache[key] = Conv2d

    return Conv2d




class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(
                grad_output,
                kernel,
                grad_kernel,
                ctx.up,
                ctx.down,
                ctx.pad,
                ctx.g_pad,
                ctx.in_size,
                ctx.out_size,
            )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    if input.device.type == "cpu":
        out = upfirdn2d_native(input, kernel, *up, *down, *pad)

    else:
        out = UpFirDn2d.apply(input, kernel, up, down, pad)

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)



class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input.contiguous(),
            gradgrad_bias,
            out,
            3,
            1,
            ctx.negative_slope,
            ctx.scale,
        )

        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)

        ctx.bias = bias is not None

        if bias is None:
            bias = empty

        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale
        )

        if not ctx.bias:
            grad_bias = None

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == "cpu":
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return (
                F.leaky_relu(
                    input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
                )
                * scale
            )

        else:
            return F.leaky_relu(input, negative_slope=0.2) * scale

    else:
        return FusedLeakyReLUFunction.apply(
            input.contiguous(), bias, negative_slope, scale
        )

def hinge_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = - logits_fake.mean() * 2 if logits_real is None else F.relu(1. + logits_fake).mean() 
    loss_real = 0 if logits_real is None else F.relu(1. - logits_real).mean()
    
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = F.softplus(-logits_fake).mean() * 2 if logits_real is None else F.softplus(logits_fake).mean()
    loss_real = 0 if logits_real is None else F.softplus(-logits_real).mean()
    
    return 0.5 * (loss_real + loss_fake)


def least_square_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = logits_fake.pow(2).mean() * 2 if logits_real is None else (1 + logits_fake).pow(2).mean()
    loss_real = 0 if logits_real is None else (1 - logits_real).pow(2).mean() 
    
    return 0.5 * (loss_real + loss_fake)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(self, num_features: int,
                 logdet: Optional[bool] = False,
                 affine: Optional[bool] = True,
                 allow_reverse_init: Optional[bool] = False) -> None:
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input: torch.FloatTensor) -> None:
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input: torch.FloatTensor, reverse: Optional[bool] = False) -> Union[torch.FloatTensor, Tuple]:
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output: torch.FloatTensor) -> torch.FloatTensor:
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
            
        return h


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]

        kernel /= kernel.sum()

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None
        
        self.scale = 1 / sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None

        self.activation = activation

        self.scale = (1 / sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel, out_channel, 
                kernel_size, padding=self.padding, 
                stride=stride, bias=bias and not activate
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class StyleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / sqrt(2)

        return out


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, use_actnorm: bool = False) -> None:
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(weights_init)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        """Standard forward."""
        return self.main(input)


class StyleDiscriminator(nn.Module):
    def __init__(self, size=256, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(log2(size))
        in_channel = channels[size]

        blocks = [ConvLayer(3, channels[size], 1)]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            blocks.append(StyleBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.blocks = nn.Sequential(*blocks)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, x):
        out = self.blocks(x)
        batch, channel, height, width = out.shape

        group = min(batch, self.stddev_group)
        group = batch//(batch//group)
        
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)
             
        return out.squeeze()


class DummyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class VQLPIPS(nn.Module):
    def __init__(self, codebook_weight: float = 1.0,
                 loglaplace_weight: float = 1.0,
                 loggaussian_weight: float = 1.0,
                 perceptual_weight: float = 1.0) -> None:
        
        super().__init__()
        self.perceptual_loss = lpips.LPIPS(net="vgg", verbose=False)

        self.codebook_weight = codebook_weight 
        self.loglaplace_weight = loglaplace_weight 
        self.loggaussian_weight = loggaussian_weight
        self.perceptual_weight = perceptual_weight 

    def forward(self, codebook_loss: torch.FloatTensor, inputs: torch.FloatTensor, reconstructions: torch.FloatTensor, optimizer_idx: int,
                global_step: int, batch_idx: int, last_layer: Optional[nn.Module] = None, split: Optional[str] = "train") -> Tuple:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()       

        loglaplace_loss = (reconstructions - inputs).abs().mean()
        loggaussian_loss = (reconstructions - inputs).pow(2).mean()
        perceptual_loss = self.perceptual_loss(inputs*2-1, reconstructions*2-1).mean()

        nll_loss = self.loglaplace_weight * loglaplace_loss + self.loggaussian_weight * loggaussian_loss + self.perceptual_weight * perceptual_loss
        loss = nll_loss + self.codebook_weight * codebook_loss
        
        log = {"{}/total_loss".format(split): loss.clone().detach(),
               "{}/quant_loss".format(split): codebook_loss.detach(),
               "{}/rec_loss".format(split): nll_loss.detach(),
               "{}/loglaplace_loss".format(split): loglaplace_loss.detach(),
               "{}/loggaussian_loss".format(split): loggaussian_loss.detach(),
               "{}/perceptual_loss".format(split): perceptual_loss.detach()
               }
        
        return loss, log


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start: int = 0,
                 disc_loss: str = 'vanilla',
                 disc_params: Optional[OmegaConf] = dict(),
                 codebook_weight: float = 1.0,
                 loglaplace_weight: float = 1.0,
                 loggaussian_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 adversarial_weight: float = 1.0,
                 use_adaptive_adv: bool = False,
                 r1_gamma: float = 10,
                 do_r1_every: int = 16) -> None:
        
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "least_square"], f"Unknown GAN loss '{disc_loss}'."
        self.perceptual_loss = lpips.LPIPS(net="vgg", verbose=False)

        self.codebook_weight = codebook_weight 
        self.loglaplace_weight = loglaplace_weight 
        self.loggaussian_weight = loggaussian_weight
        self.perceptual_weight = perceptual_weight 

        self.discriminator = StyleDiscriminator(**disc_params)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "least_square":
            self.disc_loss = least_square_d_loss

        self.adversarial_weight = adversarial_weight
        self.use_adaptive_adv = use_adaptive_adv
        self.r1_gamma = r1_gamma
        self.do_r1_every = do_r1_every

    def calculate_adaptive_factor(self, nll_loss: torch.FloatTensor,
                                  g_loss: torch.FloatTensor, last_layer: nn.Module) -> torch.FloatTensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        
        adapt_factor = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        adapt_factor = adapt_factor.clamp(0.0, 1e4).detach()

        return adapt_factor

    def forward(self, codebook_loss: torch.FloatTensor, inputs: torch.FloatTensor, reconstructions: torch.FloatTensor, optimizer_idx: int,
                global_step: int, batch_idx: int, last_layer: Optional[nn.Module] = None, split: Optional[str] = "train") -> Tuple:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()       
        
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            loglaplace_loss = (reconstructions - inputs).abs().mean()
            loggaussian_loss = (reconstructions - inputs).pow(2).mean()
            perceptual_loss = self.perceptual_loss(inputs*2-1, reconstructions*2-1).mean()

            nll_loss = self.loglaplace_weight * loglaplace_loss + self.loggaussian_weight * loggaussian_loss + self.perceptual_weight * perceptual_loss
        
            logits_fake = self.discriminator(reconstructions)
            g_loss = self.disc_loss(logits_fake)
            
            try:
                d_weight = self.adversarial_weight
                
                if self.use_adaptive_adv:
                    d_weight *= self.calculate_adaptive_factor(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = 1 if global_step >= self.discriminator_iter_start else 0
            loss = nll_loss + disc_factor * d_weight * g_loss + self.codebook_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.clone().detach(),
                   "{}/quant_loss".format(split): codebook_loss.detach(),
                   "{}/rec_loss".format(split): nll_loss.detach(),
                   "{}/loglaplace_loss".format(split): loglaplace_loss.detach(),
                   "{}/loggaussian_loss".format(split): loggaussian_loss.detach(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach(),
                   "{}/g_loss".format(split): g_loss.detach(),
                   }

            if self.use_adaptive_adv:
                log["{}/d_weight".format(split)] = d_weight.detach()
            
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            disc_factor = 1 if global_step >= self.discriminator_iter_start else 0
            do_r1 = self.training and bool(disc_factor) and batch_idx % self.do_r1_every == 0

            logits_real = self.discriminator(inputs.requires_grad_(do_r1))
            logits_fake = self.discriminator(reconstructions.detach())
            
            d_loss = disc_factor * self.disc_loss(logits_fake, logits_real)
            if do_r1:
                with conv2d_gradfix.no_weight_gradients():
                    gradients, = torch.autograd.grad(outputs=logits_real.sum(), inputs=inputs, create_graph=True)

                gradients_norm = gradients.square().sum([1,2,3]).mean()
                d_loss += self.r1_gamma * self.do_r1_every * gradients_norm/2

            log = {"{}/disc_loss".format(split): d_loss.detach(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean(),
                   }

            if do_r1:
                log["{}/r1_reg".format(split)] = gradients_norm.detach()
            
            return d_loss, log