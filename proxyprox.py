import torch
from torch.optim.optimizer import Optimizer, required


class ProxyProx(Optimizer):
    def __init__(self, params, reg=1, lr_in=0.1, lr=1, momentum_in=0, momentum_out=0, weight_decay=0,
                 momentum_estim=0, l2=False):

        defaults = dict(reg=reg, lr_in=lr_in, lr=lr,
                        momentum_in=momentum_in, momentum_out=momentum_out,
                        weight_decay=weight_decay, momentum_estim=momentum_estim,
                        l2=l2)
        super(ProxyProx, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum_out']
            momentum_estim = group['momentum_estim']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'out_grad' not in state:
                    state['out_grad'] = 1. * p.grad.data
                state['out_grad'].mul_(momentum_estim)
                state['out_grad'].add_(p.grad.data, alpha=1 - momentum_estim)
                if 'anchor' in state:
                    d_p = state['anchor'].data - p.data
                    p.add_(d_p)
                    if momentum != 0:
                        if 'momentum_buffer_out' not in state:
                            buf = state['momentum_buffer_out'] = -lr * d_p
                        else:
                            buf = state['momentum_buffer_out']
                            buf.mul_(momentum)
                            buf.add_(d_p, alpha=-lr)
                        p.add_(buf)
                    else:
                        p.add_(d_p, alpha=-lr)

                    if not group['l2'] and group['weight_decay'] != 0:
                        p.mul_(1 - lr * group['weight_decay'])                
                state['anchor'] = torch.clone(p).detach()
                if group['momentum_in'] > 0:
                    state['momentum_buffer_in'] = torch.zeros_like(p)
        return loss

    @torch.no_grad()
    def inner_step(self, closure=None, prev_optimizer=required):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            lr = group['lr_in']
            momentum = group['momentum_in']
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
               
                d_p = p.grad.data
                if group['l2'] and group['weight_decay'] != 0:
                    d_p.add_(p, alpha=group['weight_decay'])
                prev_d_p = prev_p.grad.data
                d_p.sub_(prev_d_p)
                d_p.add_(state['out_grad'])
                d_p.add_(p.data - prev_p.data, alpha=group['reg'])
                   
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer_in' not in state:
                        buf = state['momentum_buffer_in'] = -lr * d_p
                    else:
                        buf = state['momentum_buffer_in']
                        buf.mul_(momentum)
                        buf.add_(d_p, alpha=-lr)
                    p.add_(buf)
                else:
                    p.add_(d_p, alpha=-lr)
        return loss
