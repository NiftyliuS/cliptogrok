import torch


class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.sign(), alpha=-group['lr'])
