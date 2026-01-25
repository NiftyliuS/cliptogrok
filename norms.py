"""
Weight norm utilities for accelerated generalization (Clip to Grok).

Two functions:
- project_to_sphere: One-time init, normalizes all weight rows to max_norm
- clip_weight_norms: Post-step, clips decoder weights only, skips embeddings/head

Usage:
    from cliptogrok import project_to_sphere, clip_weight_norms

    model = YourModel()
    project_to_sphere(model, max_norm=2.0)

    for step in range(steps):
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            clip_weight_norms(model, max_norm=2.0)

Paper: "Clip to Grok: Weight Norm Clipping for Accelerated Generalization"
GitHub: https://github.com/NiftyliuS/cliptogrok
"""

import torch


def project_to_sphere(model, max_norm=2.0):
    """
    Normalize all weight rows to exactly max_norm (L2).
    Call once before training to align all layers to uniform scale.
    """
    for name, param in model.named_parameters():
        norm = param.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        param.mul_(max_norm / norm)


def clip_weight_norms(model, max_norm=2.0, skip_patterns=['token_embeddings', 'head']):
    """
    Clip weight rows exceeding max_norm (L2). Rows below threshold unchanged.
    Call after each optimizer.step(). Skips embeddings/head by default.
    """
    for name, param in model.named_parameters():
        if any(p in name for p in skip_patterns):
            continue
        norm = param.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(norm, max=max_norm) / norm
        param.mul_(scale)
