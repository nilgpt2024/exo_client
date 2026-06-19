import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(logits, targets, lengths=None):
    """交叉熵损失函数"""
    if lengths is not None:
        # 创建掩码
        batch_size, seq_len = targets.shape
        mask = torch.arange(seq_len).expand(batch_size, seq_len) < lengths.unsqueeze(1)

        # 应用掩码
        logits_masked = logits[mask]
        targets_masked = targets[mask]

        return F.cross_entropy(logits_masked, targets_masked)
    else:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def length_masked_ce_loss(model, inputs, targets, lengths):
    """长度掩码的交叉熵损失"""
    logits = model(inputs)
    return cross_entropy_loss(logits, targets, lengths)