import torch
import torch.nn as nn


class IdentityBlock(nn.Module):
    """占位符块，用于分片范围外的层"""

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # 如果use_cache为True且存在past_key_values参数，需要正确处理KV缓存
        use_cache = kwargs.get('use_cache', False)
        past_key_values = kwargs.get('past_key_values', None)
        
        if use_cache and past_key_values is not None:
            # 返回输入和原始的past_key_values，保持KV缓存链完整
            return x, past_key_values
        else:
            # 正常返回输入
            return x