"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        Args:
            params: iterable of parameters of type `needle.nn.Parameter` to optimize
            lr: (*float*) - learning rate
            momentum: (*float*) - momentum factor
            weight_decay: (*float*) - weight decay (L2 penalty)
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        self.u = {p: self.momentum * self.u.get(p, ndl.init.zeros(*p.shape)).data + (1 - self.momentum) * (p.grad.data + self.weight_decay * p.data) for p in self.params}
        for p in self.params:
            p.data -= self.lr * self.u[p].data


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        """
        Args:
            params: iterable of parameters of type `needle.nn.Parameter` to optimize
            lr: (*float*) - learning rate
            beta1: (*float*) - coefficient used for computing running average of gradient
            beta2: (*float*) - coefficient used for computing running average of square of gradient
            eps: (*float*) - term added to the denominator to improve numerical stability
            bias_correction: - whether to use bias correction for $u, v$
            weight_decay: (*float*) - weight decay (L2 penalty)
        """
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        self.m = {p: self.beta1 * self.m.get(p, ndl.init.zeros(*p.shape)).data + (1 - self.beta1) * (p.grad.data + self.weight_decay * p.data) for p in self.params}
        self.v = {p: self.beta2 * self.v.get(p, ndl.init.zeros(*p.shape)).data + (1 - self.beta2) * (p.grad.data + self.weight_decay * p.data) ** 2 for p in self.params}
        m_hat = {p: self.m[p].data / (1 - self.beta1 ** self.t) for p in self.params}
        v_hat = {p: self.v[p].data / (1 - self.beta2 ** self.t) for p in self.params}
        for p in self.params:
            p.data -= self.lr * m_hat[p].data / (v_hat[p].data ** 0.5 + self.eps)
