import torch

def adaptive_update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, min_ratio=0.1):
        # step weight decay
        p.data.mul_(1. - lr * wd)

        # weight update
        update = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1).sign_()

        # agreement-based LR scaling
        agreement = (p.data.sign() == update).float().mean().item()
        lr_scale = min_ratio + (1.0 - min_ratio) * agreement

        p.add_(update, alpha=-lr * lr_scale)

        # decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)
    # class

class SignSGDV2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, beta=0.9):
        super().__init__(params, dict(lr=lr, beta=beta))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['ema'] = torch.zeros_like(p.data)

                ema = state['ema']
                ema.mul_(beta).add_(p.grad, alpha=1 - beta)
                p.data.add_(ema.sign(), alpha=-group['lr'])


class EMASignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, beta=0.9):
        super().__init__(params, dict(lr=lr, beta=beta))
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['ema'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue

                ema = self.state[p]['ema']
                ema.mul_(beta).add_(p.grad, alpha=1 - beta)
                p.data.add_(ema.sign(), alpha=-group['lr'])

class AdaptiveEMASignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, beta=0.9, min_ratio=0.1):
        super().__init__(params, dict(lr=lr, beta=beta, min_ratio=min_ratio))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta = group['beta']
            min_ratio = group['min_ratio']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['ema'] = torch.zeros_like(p.data)

                ema = state['ema']
                ema.mul_(beta).add_(p.grad, alpha=1 - beta)

                agreement = (p.data.sign() == ema.sign()).float().mean()
                lr_scale = min_ratio + (1.0 - min_ratio) * agreement
                p.data.add_(ema.sign(), alpha=-group['lr'] * lr_scale)

class AdaptiveEMASignSGDV2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, beta=0.9, min_ratio=0.1):
        super().__init__(params, dict(lr=lr, beta=beta, min_ratio=min_ratio))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta = group['beta']
            min_ratio = group['min_ratio']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['ema'] = torch.zeros_like(p.data)

                ema = state['ema']
                ema.mul_(beta).add_(p.grad, alpha=1 - beta)

                # agreement between weights and EMA direction
                agreement = (p.data.sign() == ema.sign()).float().mean().item()
                lr_scale = min_ratio + (1.0 - min_ratio) * max(0.0, 2.0 * (agreement - 0.5))

                p.data.add_(ema.sign(), alpha=-group['lr'] * lr_scale)

class SignSGDBuffer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        super().__init__(params, dict(lr=lr, beta=beta, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['ema'] = torch.zeros_like(p)

                g = p.grad
                ema = state['ema']

                # Scale gradient to [-1, 1]
                scaled_g = g / g.abs().max().clamp(min=eps)

                # Blend
                ema.mul_(group['beta']).add_(scaled_g, alpha=1 - group['beta'])

                # Scale output to [-1, 1]
                scaled = ema / ema.abs().max().clamp(min=eps)

                p.data.add_(scaled, alpha=-group['lr'])