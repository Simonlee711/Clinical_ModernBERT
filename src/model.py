import math
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM

class LowPrecisionLayerNorm(nn.LayerNorm):
    def forward(self, input):
        orig_dtype = input.dtype
        if input.dtype == torch.float32:
            input = input.to(torch.bfloat16)
        out = super().forward(input)
        return out.to(orig_dtype)

class MosaicBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dropout_prob = 0.0
        self.attention_probs_dropout_prob = 0.0
        self.flash_attention = True
        self.alibi = True
        self.gated_linear_units = True
        self.use_low_precision_layernorm = True
        self.rope_theta = kwargs.get("rope_theta", 10000)
        self.context_length = kwargs.get("context_length", 1024)

class MosaicBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config: MosaicBertConfig):
        super().__init__(config)
        if config.use_low_precision_layernorm:
            for name, module in self.named_modules():
                if isinstance(module, nn.LayerNorm):
                    new_ln = LowPrecisionLayerNorm(module.normalized_shape, module.eps, module.elementwise_affine)
                    with torch.no_grad():
                        new_ln.weight.copy_(module.weight)
                        if module.bias is not None:
                            new_ln.bias.copy_(module.bias)
                    parent = self
                    *path, last = name.split('.')
                    for p in path:
                        parent = getattr(parent, p)
                    setattr(parent, last, new_ln)
    
    def forward(self, *args, **kwargs):
        if 'num_items_in_batch' in kwargs:
            kwargs.pop('num_items_in_batch')
        return super().forward(*args, **kwargs)

class StableAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, clipping_threshold=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, clipping_threshold=clipping_threshold)
        super(StableAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                clipping_threshold = group['clipping_threshold']
                lr = group['lr']
                weight_decay = group['weight_decay']
                eps = group['eps']

                state['step'] += 1

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / bias_correction1
                update_norm = update.norm()
                denom_norm = denom.norm()

                if update_norm > clipping_threshold * denom_norm:
                    update.mul_(clipping_threshold * denom_norm / update_norm)

                p.data.addcdiv_(update, denom, value=-lr / math.sqrt(bias_correction2))
        return loss
