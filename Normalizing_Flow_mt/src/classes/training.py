import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from logging_setup_configs import setup_logging
import math

def linear_warmup_then_cosine(optimizer, step, warmup_steps, base_lr, total_steps):     # not used relict
    """Linear warmup to base_lr, then cosine decay to 0."""
    if step < warmup_steps:
        scale = (step + 1) / max(1, warmup_steps)
    else:
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        scale = 0.5 * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg['lr'] = base_lr * scale

def warmup_s_scale(model, epoch, total_epochs, max_scale=1.5):              # not used relict
    """Cosine warmup for s_scale inside AffineCoupling modules."""
    alpha = (epoch + 1) / max(total_epochs, 1)
    scale = max_scale * 0.5 * (1.0 - math.cos(math.pi * alpha))
    for m in model.modules():
        if hasattr(m, "s_scale"):
            # buffer or attribute
            if isinstance(m.s_scale, torch.Tensor):
                m.s_scale = torch.tensor(scale, device=m.s_scale.device)
            else:
                setattr(m, "s_scale", scale)

def grad_global_norm(parameters):               #not used relict
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            n = p.grad.data.norm(2)
            total += n.item() ** 2
    return total ** 0.5

def assert_finite(tensor, name="tensor"):               #not used relict
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def train_epoch(train_loader, model, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for x_batch, w_batch in train_loader:
        x_batch = x_batch.to(device)
        w_batch = w_batch.to(device)
        log_px = model.log_prob(x_batch)
        loss = (-log_px * w_batch).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    avg_nll = total_loss / max(total_batches, 1)
    
    return avg_nll

def train_epoch_no_weights(train_loader, model, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for x_batch, w_batch  in train_loader:
        x_batch = x_batch.to(device)
        log_px = model.log_prob(x_batch)
        loss = (-log_px).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    avg_nll = total_loss / max(total_batches, 1)
    
    return avg_nll 


def val_epoch(val_loader, model, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for x_batch, w_batch in val_loader:
            x_batch = x_batch.to(device)
            w_batch = w_batch.to(device)
            log_px = model.log_prob(x_batch)
            loss = (-log_px * w_batch).sum()
            total_loss += loss.item()
            total_batches += 1
    avg_nll = total_loss / max(total_batches, 1)
    return avg_nll

def val_epoch_no_weights(val_loader, model, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for x_batch, w_batch in val_loader:
            x_batch = x_batch.to(device)
            log_px = model.log_prob(x_batch)
            loss = (-log_px).sum()
            total_loss += loss.item()
            total_batches += 1
    avg_nll = total_loss / max(total_batches, 1)
    return avg_nll