import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig


class MaskAwareClassifier(nn.Module):
    def __init__(self, config: BertConfig, n_classes: int):
        super(MaskAwareClassifier, self).__init__()

        self.classifier = nn.Linear(config.hidden_size, n_classes)

    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
            x:
            masks (Optional[torch.Tensor], optional): (bsz, n_queries)

        Returns:

        """
        x = self.classifier.forward(x)
        if masks is not None:
            x = torch.masked_fill(x, masks.unsqueeze(-1), 0)  # mask fill一些不可导的值
        return x


class MaskAwareSSNFuse(nn.Module):
    """fuse bert embeddings with query embeddings"""

    def __init__(self, config: BertConfig):
        super(MaskAwareSSNFuse, self).__init__()
        self.hidden_dim = config.hidden_size
        self.W = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_mask: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None):
        """

        Args:
            query:
            key:
            key_mask:
            query_mask (Optional[torch.Tensor], optional): (bsz, n_queries)

        Returns:

        """
        key_embed = key.unsqueeze(1).expand(-1, query.size(1), -1, -1)
        query_embed = query.unsqueeze(2).expand(-1, -1, key.size(-2), -1)

        fuse = torch.cat([key_embed, query_embed], dim=-1)
        x = self.W(fuse)
        x = self.v(torch.tanh(x)).squeeze(-1)
        key_mask = key_mask.unsqueeze(1).expand(-1, x.size(1), -1)
        x[key_mask] = -1e25

        if query_mask is not None:
            x = torch.masked_fill(x, query_mask.unsqueeze(-1), 0)  # mask fill一些不可导的值

        # x = F.sigmoid(x)
        x = F.softmax(x, dim=-1)

        return x

# class MaskAwarePIQNBoundaryPredictor(nn.Module):
#     def __init__(self, config: BertConfig):
#         super(MaskAwarePIQNBoundaryPredictor, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.token_embedding_linear = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
#         self.entity_embedding_linear = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
#         self.boundary_predictor = nn.Linear(self.hidden_size, 1)
#
#     def forward(self, entity_embedding: torch.Tensor, token_embedding: torch.Tensor,
#                 padding_mask: torch.Tensor, query_mask: Optional[torch.Tensor] = None):
#         # B x #ent x #token x hidden_size
#         entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(
#             entity_embedding).unsqueeze(2)
#         entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix)).squeeze(-1)
#         padding_mask = padding_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
#         entity_token_cls[padding_mask] = -10000
#
#         if query_mask is not None:
#             entity_token_cls = torch.masked_fill(entity_token_cls, query_mask.unsqueeze(-1), 0)  # mask fill一些不可导的值
#
#         entity_token_p = F.sigmoid(entity_token_cls)
#         # entity_token_p = F.softmax(entity_token_cls, dim=-1)
#
#         return entity_token_p
#
#
# class MaskAwarePIQNTypePredictor(nn.Module):
#     def __init__(self, config: BertConfig, entity_type_count: int):
#         super(MaskAwarePIQNTypePredictor, self).__init__()
#
#         self.linear = nn.Linear(config.hidden_size, config.hidden_size)
#
#         self.attn = nn.MultiheadAttention(config.hidden_size, dropout=config.hidden_dropout_prob,
#                                           num_heads=config.num_attention_heads)
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(config.hidden_size * 3, entity_type_count)
#         )
#
#     def forward(self, h_entity: torch.Tensor, h_token: torch.Tensor, p_left: torch.Tensor, p_right: torch.Tensor,
#                 padding_mask: torch.Tensor, query_mask: Optional[torch.Tensor] = None):
#         h_entity = self.linear(torch.relu(h_entity))
#
#         attn_output, _ = self.attn(h_entity.transpose(0, 1).clone(), h_token.transpose(0, 1),
#                                    h_token.transpose(0, 1), key_padding_mask=padding_mask)
#         attn_output = attn_output.transpose(0, 1)
#         h_entity += attn_output
#
#         left_token = torch.matmul(p_left, h_token)
#         right_token = torch.matmul(p_right, h_token)
#
#         h_entity = torch.cat([h_entity, left_token, right_token], dim=-1)
#
#         entity_logits = self.classifier.forward(h_entity)
#
#         if query_mask is not None:
#             entity_logits = torch.masked_fill(entity_logits, query_mask.unsqueeze(-1), 0)  # mask fill一些不可导的值
#
#         return entity_logits
