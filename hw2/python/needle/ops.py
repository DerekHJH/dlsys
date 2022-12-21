"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + array_api.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * array_api.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** array_api.float32(self.scalar)

    def gradient(self, out_grad, node):
        # TODO: self.scaler == 0, and out_grad should appear in the left or right
        lhs = node.inputs[0]
        return (self.scalar * lhs ** (self.scalar - 1)) * out_grad


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs * rhs)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / array_api.float32(self.scalar)

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        """
        Args:
            axes: A tuple of length 2 (axis1, axis2), defaults to the last two axes
        """
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, axis1=len(a.shape) - 2, axis2=len(a.shape) - 1)
        return array_api.swapaxes(a, axis1=self.axes[0], axis2=self.axes[1])

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        if self.axes is None:
            return out_grad.transpose()
        return out_grad.transpose((self.axes[0], self.axes[1]))


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return out_grad.reshape(lhs.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        axes = [-i-1 for i in range(len(self.shape)) if (i >= len(lhs.shape) or lhs.shape[-i-1] != self.shape[-i-1])]
        axes = sorted([i + len(self.shape) for i in axes])
        return out_grad.sum(axes=tuple(axes)).reshape(lhs.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        axes = self.axes
        if isinstance(self.axes, int):
            axes = [axes,]
        shape = [1 if (axes is None or i in axes) else lhs.shape[i] for i in range(len(lhs.shape))]
        return out_grad.reshape(tuple(shape)).broadcast_to(lhs.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        retl = out_grad @ rhs.transpose()
        retr = lhs.transpose() @ out_grad
        if retl.shape != lhs.shape:
            retl = retl.sum(axes=tuple(range(len(retl.shape) - len(lhs.shape))))
        if retr.shape != rhs.shape:
            retr = retr.sum(axes=tuple(range(len(retr.shape) - len(rhs.shape))))
        return retl, retr


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return out_grad / lhs


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return Tensor(node.realize_cached_data() > 0) * out_grad


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        """
        Args:
            axes: Tuple of axes to sum and take the maximum element over. 
            This uses the same conventions as `needle.ops.Summation()`
        """
        self.axes = axes

    def compute(self, Z):
        Z_max_keepdim = array_api.max(Z, axis=self.axes, keepdims=True)
        self.Z_max = Tensor(array_api.broadcast_to(Z_max_keepdim, Z.shape)) # for gradient
        Z_max = array_api.max(Z, axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - Z_max_keepdim), axis=self.axes)) + Z_max

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        axes = self.axes
        if isinstance(self.axes, int):
            axes = [axes,]
        shape = [1 if (axes is None or i in axes) else lhs.shape[i] for i in range(len(lhs.shape))]
        exp_tensor = exp(lhs - self.Z_max)
        sum_tensor = exp_tensor.sum(axes=self.axes).reshape(tuple(shape)).broadcast_to(lhs.shape)
        return out_grad.reshape(tuple(shape)) * exp_tensor / sum_tensor


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
