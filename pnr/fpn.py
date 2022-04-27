from abc import abstractmethod, ABC
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import BertConfig


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`
    """
    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):
        # Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        nn.init.uniform_(weight, -sampling_range, sampling_range)

        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def generate_boundary(max_len: int, max_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        max_len:
        max_depth:

    Returns:
        boundary matrix (torch.Tensor), (max_depth, max_position_embedding, 2), dtype=torch.int64
    """
    assert max_depth <= max_len, "Max length ({}) should be greater than max depth ({})".format(max_len, max_depth)
    left_b = torch.arange(0, max_len, dtype=torch.int64)
    right_b = left_b
    left_boundaries = [left_b[:i] for i in range(max_len, max_len - max_depth, -1)]
    right_boundaries = [right_b[i:] for i in range(max_depth)]
    left_boundaries = torch.nn.utils.rnn.pad_sequence(left_boundaries, batch_first=True)
    right_boundaries = torch.nn.utils.rnn.pad_sequence(right_boundaries, batch_first=True)
    return left_boundaries, right_boundaries


class BasePyramidFeatureNet(nn.Module, ABC):
    def __init__(self, config: BertConfig, num_layer: int, dropout=0.):
        super(BasePyramidFeatureNet, self).__init__()
        self.config = config
        self.num_layer = num_layer
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(config.hidden_size)

        self.left_boundary, self.right_boundary = generate_boundary(config.max_position_embeddings, num_layer + 1)
        self.left_boundary = nn.Parameter(self.left_boundary, requires_grad=False)
        self.right_boundary = nn.Parameter(self.right_boundary, requires_grad=False)

    @property
    @abstractmethod
    def output_feature_size(self) -> int:
        pass

    @abstractmethod
    def init_weights(self):
        pass


class PyramidFeatureNet(BasePyramidFeatureNet):
    def __init__(self, config: BertConfig, num_layer: int, dropout=0., position: Optional[torch.Tensor] = None):
        """

        :param config:
        :param num_layer:
        :param dropout:
        """
        super(PyramidFeatureNet, self).__init__(config, num_layer, dropout)
        self.position = position
        self.input_encoding_norm = nn.LayerNorm(config.hidden_size)

        self.combine_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 其实卷积和linear也没有什么差别吧

    @property
    def output_feature_size(self):
        return self.config.hidden_size

    def init_weights(self):
        init_linear(self.combine_layer)

    def forward(self, encoding: torch.Tensor, padding_mask: torch.Tensor, sent_lens: torch.Tensor):
        """

        Args:
            encoding:
            padding_mask (torch.Tensor): (bsz, seq_len)
            sent_lens (torch.Tensor): 需要在cpu上

        Returns:
            layer_feature [(bsz, len_seq, C)]; layer_mask [(bsz, len_seq)], true for mask;
                feature (bsz, candidates, C); feature_mask (bsz, candidates);
                left_boundary (bsz, candidates); right_boundary (bsz, candidates)

        """
        bsz, len_seq, _ = encoding.size()
        layer_feature = []
        layer_mask = []
        encoding = self.input_encoding_norm.forward(encoding)

        if self.position is not None:
            encoding = encoding + self.position[:len_seq].unsqueeze(0).expand(bsz, -1, -1)

        num_layer = min(self.num_layer, encoding.shape[1])

        for i in range(num_layer):
            if i == 0:
                mask = padding_mask
            else:
                encoding = torch.cat([encoding[:, :-1], encoding[:, 1:]], dim=-1)  # (B, T, 2*H)
                encoding = self.combine_layer.forward(encoding)  # (B, T, H)

                mask = padding_mask[:, i:]
            layer_mask.append(mask)

            encoding = self.norm.forward(encoding)
            encoding = self.dropout_layer.forward(encoding)

            layer_feature.append(encoding)

        feature = torch.cat(layer_feature, dim=1)
        feature_mask = torch.cat(layer_mask, dim=1)

        def make_boundary(boundary_matrix: torch.Tensor):
            return torch.cat([boundary_matrix[i, :len_seq - i]
                              for i in range(len(layer_feature))], dim=-1).expand(bsz, -1)

        left_boundary = make_boundary(self.left_boundary)
        right_boundary = make_boundary(self.right_boundary)

        return (layer_feature, layer_mask), (feature, feature_mask), (left_boundary, right_boundary)


class BiPyramidFeatureNet(BasePyramidFeatureNet):
    def __init__(self, config: BertConfig, num_layer: int, dropout=0., position: Optional[torch.Tensor] = None):
        """

        :param config:
        :param num_layer:
        :param dropout:
        """
        super(BiPyramidFeatureNet, self).__init__(config, num_layer, dropout)
        self.position = position
        self.input_encoding_norm = nn.LayerNorm(config.hidden_size)

        self.combine_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 其实卷积和linear也没有什么差别吧

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=2,
            padding=1,
        )

    @property
    def output_feature_size(self):
        return self.config.hidden_size * 2

    def init_weights(self):
        init_linear(self.combine_layer)

    def split_layer(self, encoding: torch.Tensor) -> torch.Tensor:
        return self.conv.forward(encoding.transpose(1, 2)).transpose(1, 2)

    def forward(self, encoding: torch.Tensor, padding_mask: torch.Tensor, sent_lens: torch.Tensor):
        """

        Args:
            encoding:
            padding_mask (torch.Tensor): (bsz, seq_len)
            sent_lens (torch.Tensor): 需要在cpu上

        Returns:

        """
        bsz, len_seq, _ = encoding.size()
        layer_features = []
        inv_layer_features = []
        layer_mask = []
        encoding = self.input_encoding_norm.forward(encoding)

        if self.position is not None:
            encoding = encoding + self.position[:len_seq].unsqueeze(0).expand(bsz, -1, -1)

        num_layer = min(self.num_layer, encoding.shape[1])

        for i in range(num_layer):
            if i == 0:
                mask = padding_mask
            else:
                encoding = torch.cat([encoding[:, :-1], encoding[:, 1:]], dim=-1)  # (B, T, 2*H)
                encoding = self.combine_layer.forward(encoding)  # (B, T, H)

                mask = padding_mask[:, i:]
            layer_mask.append(mask)

            encoding = self.norm.forward(encoding)
            encoding = self.dropout_layer.forward(encoding)

            layer_features.append(encoding)

        for i in range(num_layer, 0, -1):
            if i == num_layer:
                encoding = torch.zeros_like(encoding)
            else:
                encoding = self.split_layer(torch.cat([
                    encoding, layer_features[i]
                ], dim=-1))

                encoding = self.norm.forward(encoding)
                encoding = self.dropout_layer.forward(encoding)

            inv_layer_features.append(encoding)

        layer_features = [
            torch.cat([feat, inv_feat], dim=-1) for feat, inv_feat in zip(layer_features, reversed(inv_layer_features))
        ]

        features = torch.cat(layer_features, dim=1)
        feature_mask = torch.cat(layer_mask, dim=1)

        def make_boundary(boundary_matrix: torch.Tensor):
            return torch.cat(
                [boundary_matrix[i, :len_seq - i] for i in range(len(layer_features))], dim=-1
            ).expand(bsz, -1)

        left_boundary = make_boundary(self.left_boundary)
        right_boundary = make_boundary(self.right_boundary)

        return (layer_features, layer_mask), (features, feature_mask), (left_boundary, right_boundary)


class PyramidFeatureLSTMNet(BasePyramidFeatureNet):
    def __init__(self, config: BertConfig, num_layer: int, dropout=0.):
        """

        :param config:
        :param num_layer:
        :param dropout:
        """
        super(PyramidFeatureLSTMNet, self).__init__(config, num_layer, dropout)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=1,
                            bias=True, batch_first=True, dropout=dropout, bidirectional=True)

        self.combine_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 其实卷积和linear也没有什么差别吧

    @property
    def output_feature_size(self):
        return self.config.hidden_size

    def init_weights(self):
        init_linear(self.combine_layer)
        init_lstm(self.lstm)

    def lstm_forward(self, encoding: torch.Tensor, sent_lens: torch.Tensor) -> torch.Tensor:
        # sentence长度为0会发生错误，这里强制修改为1，并不会影响结果因为长度为0的都会是mask
        sent_lens = torch.maximum(sent_lens, torch.as_tensor([1]))
        encoding = torch.nn.utils.rnn.pack_padded_sequence(encoding, sent_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm.forward(encoding, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out

    def forward(self, encoding: torch.Tensor, padding_mask: torch.Tensor, sent_lens: torch.Tensor):
        """

        Args:
            encoding:
            padding_mask (torch.Tensor): (bsz, seq_len)
            sent_lens (torch.Tensor): 需要在cpu上

        Returns:
            layer_feature [(bsz, len_seq, C)]; layer_mask [(bsz, len_seq)], true for mask;
                feature (bsz, candidates, C); feature_mask (bsz, candidates);
                left_boundary (bsz, candidates); right_boundary (bsz, candidates)

        """
        bsz, len_seq, _ = encoding.size()
        layer_feature = []
        layer_mask = []

        num_layer = min(self.num_layer, encoding.shape[1])

        for i in range(num_layer):
            if i == 0:
                mask = padding_mask
            else:
                encoding = torch.cat([encoding[:, :-1], encoding[:, 1:]], dim=-1)  # (B, T, 2*H)
                encoding = self.combine_layer.forward(encoding)  # (B, T, H)

                mask = padding_mask[:, i:]

                # 一般而言encoding已经建模了token级别的相互关系了，不需要再层内lstm了
                encoding = self.lstm_forward(encoding, sent_lens - i)

            encoding = self.norm.forward(encoding)
            encoding = self.dropout_layer.forward(encoding)

            layer_mask.append(mask)
            layer_feature.append(encoding)

        feature = torch.cat(layer_feature, dim=1)
        feature_mask = torch.cat(layer_mask, dim=1)

        def make_boundary(boundary_matrix: torch.Tensor):
            return torch.cat([boundary_matrix[i, :len_seq - i]
                              for i in range(len(layer_feature))], dim=-1).expand(bsz, -1)

        left_boundary = make_boundary(self.left_boundary)
        right_boundary = make_boundary(self.right_boundary)

        return (layer_feature, layer_mask), (feature, feature_mask), (left_boundary, right_boundary)


class BiPyramidFeatureLSTMNet(BasePyramidFeatureNet):
    def __init__(self, config: BertConfig, num_layer: int, dropout=0.):
        """

        :param config:
        :param num_layer:
        :param dropout:
        """
        super(BiPyramidFeatureLSTMNet, self).__init__(config, num_layer, dropout)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=1,
                            bias=True, batch_first=True, dropout=dropout, bidirectional=True)

        self.combine_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 其实卷积和linear也没有什么差别吧

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=2,
            padding=1,
        )

    @property
    def output_feature_size(self):
        return self.config.hidden_size * 2

    def init_weights(self):
        init_linear(self.combine_layer)
        init_lstm(self.lstm)

    def split_layer(self, encoding: torch.Tensor) -> torch.Tensor:
        return self.conv.forward(encoding.transpose(1, 2)).transpose(1, 2)

    def lstm_forward(self, encoding: torch.Tensor, sent_lens: torch.Tensor) -> torch.Tensor:
        """

        Args:
            encoding:
            sent_lens:

        Returns:

        """
        # sentence长度为0会发生错误，这里强制修改为1，并不会影响结果因为长度为0的都会是mask
        sent_lens = torch.maximum(sent_lens, torch.as_tensor([1]))
        encoding = torch.nn.utils.rnn.pack_padded_sequence(encoding, sent_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm.forward(encoding, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out

    def forward(self, encoding: torch.Tensor, padding_mask: torch.Tensor, sent_lens: torch.Tensor):
        """

        Args:
            encoding:
            padding_mask (torch.Tensor): (bsz, seq_len)
            sent_lens (torch.Tensor): 需要在cpu上

        Returns:

        """
        bsz, len_seq, _ = encoding.size()
        layer_features = []
        inv_layer_features = []
        layer_mask = []

        num_layer = min(self.num_layer, encoding.shape[1])

        for i in range(num_layer):
            if i == 0:
                mask = padding_mask
            else:
                encoding = torch.cat([encoding[:, :-1], encoding[:, 1:]], dim=-1)  # (B, T, 2*H)
                encoding = self.combine_layer.forward(encoding)  # (B, T, H)

                mask = padding_mask[:, i:]
            layer_mask.append(mask)

            encoding = self.norm.forward(encoding)

            encoding = self.lstm_forward(encoding, sent_lens - i)

            encoding = self.norm.forward(encoding)
            encoding = self.dropout_layer.forward(encoding)

            layer_features.append(encoding)

        for i in range(num_layer, 0, -1):
            if i == num_layer:
                encoding = torch.zeros_like(encoding)
            else:
                encoding = self.split_layer(torch.cat([
                    encoding, layer_features[i]
                ], dim=-1))
                encoding = self.norm.forward(encoding)

                encoding = self.lstm_forward(encoding, sent_lens - i + 1)

                encoding = self.norm.forward(encoding)
                encoding = self.dropout_layer.forward(encoding)

            inv_layer_features.append(encoding)

        layer_features = [
            torch.cat([feat, inv_feat], dim=-1) for feat, inv_feat in zip(layer_features, reversed(inv_layer_features))
        ]

        features = torch.cat(layer_features, dim=1)
        feature_mask = torch.cat(layer_mask, dim=1)

        def make_boundary(boundary_matrix: torch.Tensor):
            return torch.cat([boundary_matrix[i, :len_seq - i]
                              for i in range(len(layer_features))], dim=-1).expand(bsz, -1)

        left_boundary = make_boundary(self.left_boundary)
        right_boundary = make_boundary(self.right_boundary)

        return (layer_features, layer_mask), (features, feature_mask), (left_boundary, right_boundary)


def unit_test():
    config = BertConfig()
    net = BiPyramidFeatureNet(config=config, num_layer=8)

    encoding = torch.randn((2, 10, 768), requires_grad=True)
    sent_lens = torch.as_tensor([10, 4])
    padding_mask = torch.ones((2, 10), dtype=torch.bool)
    padding_mask[0, :10] = 0
    padding_mask[1, :4] = 0

    (layer_feature, layer_mask), (feature, feature_mask), (left_boundary, right_boundary) = net.forward(
        encoding, padding_mask, sent_lens
    )
    print(layer_mask)
    print(feature_mask)
    print(left_boundary)
    print(right_boundary)
    print(feature.size())
    print(feature_mask.size())

    classifier = nn.Linear(net.output_feature_size, 1)
    cls = classifier.forward(feature).squeeze(-1)
    cls = torch.masked_fill(cls, feature_mask, 1e-9)
    print(cls.size())
    topk = torch.topk(cls, 10 * 10, dim=1, sorted=False)
