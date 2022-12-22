"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        """
        Applies a linear transformation to the incoming data: $y = xA^T + b$. The input shape is $(N, H_{in})$.
        The learnable `weight` of shape (`in_features`, `out_features`), whose values is initialized 
        with the Kaiming Uniform initialization with `fan_in = in_features`
        The learnable `bias` of shape (`out_features`), whose values should be initialized 
        with the Kaiming Uniform initialization with `fan_in = out_features`. 
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to `False`, the layer will not learn an additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).reshape((1, self.out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        if self.bias is None:
            return X @ self.weight
        else:
            return X @ self.weight + self.bias.broadcast_to((*X.shape[:-1], self.out_features))



class Flatten(Module):
    def forward(self, X):
        prod = 1
        for i in range(1, len(X.shape)):
            prod *= X.shape[i]
        return X.reshape((X.shape[0], prod))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the rectified linear unit function element-wise:
        $ReLU(x) = max(0, x)$.
        """
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        """
        Args:
            *modules - any number of modules of type `needle.nn.Module`
        """
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y_one_hot = init.one_hot(logits.shape[-1], y)
        return ops.summation(ops.logsumexp(logits, axes = 1) - ops.summation(logits * y_one_hot, axes = 1)) / logits.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        """
        Args:
            dim: input dimension
            eps: a value added to the denominator for numerical stability.
            momentum: the value used for the running mean and running variance computation.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype)) # the learnable weights of size `dim`, elements initialized to 1
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype)) # the learnable bias of shape `dim`, elements initialized to 0.
        self.running_mean = init.zeros(dim, requires_grad=True, device=device, dtype=dtype) # the running mean used at evaluation time, elements initialized to 0.
        self.running_var = init.ones(dim, requires_grad=True, device=device, dtype=dtype) # the running (unbiased) variance used at evaluation time, elements initialized to 1. 

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            Ex = ops.summation(x, axes = 0) / x.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * Ex
            Ex = Ex.reshape((1, x.shape[1])).broadcast_to(x.shape)
            Varx = ops.summation((x - Ex) ** 2, axes = 0) / x.shape[0]
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Varx
            Varx = Varx.reshape((1, x.shape[1])).broadcast_to(x.shape)
            return self.weight.broadcast_to(x.shape) * (x - Ex) / (Varx + self.eps) ** 0.5 + self.bias.broadcast_to(x.shape)
        else:
            return (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
        


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        """
        $y = w \circ \frac{x_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b$

        where $\textbf{E}[x]$ denotes the empirical mean of the inputs, 
        $\textbf{Var}[x]$ denotes their empirical variance (not that here we are using the "unbiased" 
        estimate of the variance, i.e., dividing by $N$ rather than by $N-1$), 
        and $w$ and $b$ denote learnable scalar weights and biases respectively.  
        Note you can assume the input to this layer is a 2D tensor, 
        with batches in the first dimension and features on the second.
        Args:
            dim: number of channels
            eps: a value added to the denominator for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype)) # the learnable weights of size `dim`, elements initialized to 1
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype)) # the learnable bias of shape `dim`, elements initialized to 0.

    def forward(self, x: Tensor) -> Tensor:
        Ex = (ops.summation(x, axes = 1) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        Varx = (ops.summation((x - Ex) ** 2, axes = 1) / self.dim).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * (x - Ex) / (Varx + self.eps) ** 0.5 + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p = 0.5):
        """
        Args:
            p: the probability of an element to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p = (1 - self.p), dtype = "float32")
            x = x / (1 - self.p) * mask
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        """
        Args:
            fn: module of type `needle.nn.Module`
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x



