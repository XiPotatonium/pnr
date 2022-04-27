import copy
import math
import warnings

import torch
import torch.nn.functional as F
import torchsnooper
from torch import nn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x: torch.Tensor, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def repeat_modules(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def make_position_embedding(max_embedding: int, embedding_dim: int, temperature=10000.0):
    """

    :param max_embedding: 需要多少个位置
    :param embedding_dim: 位置编码的size
    :param temperature:
    :return: (bs, num_embed, d_embed)
    """
    dtype = torch.float
    pos = torch.arange(max_embedding, dtype=dtype) / max_embedding * 2 * math.pi

    dim_t = torch.arange(embedding_dim, dtype=dtype)
    dim_t = temperature ** (2 * (dim_t // 2) / embedding_dim)

    pos = pos.unsqueeze(-1) / dim_t
    pos[:, 0::2].sin_()
    pos[:, 1::2].cos_()

    pos = pos.detach()
    pos.requires_grad = False
    return pos


def make_sent_position_embedding(embedding_matrix: torch.Tensor, sent_lens: torch.Tensor):
    """

    :param embedding_matrix: (num_embedding, embedding_dim)
    :param sent_lens: (bs)
    :return: (bs, num_embed, d_embed)
    """
    pos_embed = [embedding_matrix[:sent_len] for sent_len in sent_lens]
    pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True)

    return pos_embed
