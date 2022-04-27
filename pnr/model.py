import math
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .predictor import MaskAwareClassifier, MaskAwareSSNFuse
from .fpn import BasePyramidFeatureNet, PyramidFeatureNet, BiPyramidFeatureNet, PyramidFeatureLSTMNet, \
    BiPyramidFeatureLSTMNet
from .dec import TransformerDecoderLayer as SSNTrfDecoderLayer, TransformerDecoder as SSNTrfDecoder
from .util import MLP, make_position_embedding, make_sent_position_embedding


class MultiScaleSSN(BertPreTrainedModel):
    def __init__(
        self, config: BertConfig,
        n_classes: int, n_queries: int, dropout=0.1, pool_type="max",
        use_lstm=False, lstm_drop=0.1, use_glove=True, embed: Optional[torch.Tensor] = None, use_pos=False, pos_size=25,
        use_char_lstm=True, char_size=25, char_lstm_layers=1, char_lstm_drop=0.2,

        fpn_type="uni", fpn_pos_type: Optional[str] = None, fpn_layer=16, fpn_drop=0.1,
        use_topk_query=True, use_msf=True,
        dec_type="ssn", dec_layers=3, dec_intermediate_size=1024, dec_num_attention_heads=8,

        split_epoch=0, freeze_transformer=True, aux_loss=True
    ):
        super(MultiScaleSSN, self).__init__(config)
        self.config = config
        self.n_classes = n_classes
        self.n_queries = n_queries
        self.dropout = nn.Dropout(dropout)
        self.pool_type = pool_type
        self.use_lstm = use_lstm
        lstm_input_size = config.hidden_size
        self.use_glove = use_glove
        if use_glove:
            assert embed is not None
            self.word2vec_size = embed.size()[-1]
            self.word2vec_ebd = nn.Embedding.from_pretrained(embed, freeze=False)
            lstm_input_size += self.word2vec_size
        self.use_pos = use_pos
        if use_pos:
            lstm_input_size += pos_size
            self.pos_embedding = nn.Embedding(100, pos_size)
        self.use_char_lstm = use_char_lstm
        if use_char_lstm:
            lstm_input_size += char_size * 2
            self.char_size = char_size
            self.char_lstm = nn.LSTM(input_size=char_size, hidden_size=char_size,
                                     num_layers=char_lstm_layers, bidirectional=True,
                                     dropout=char_lstm_drop, batch_first=True)
            self.char_embedding = nn.Embedding(103, char_size)
        if use_lstm:
            self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=config.hidden_size // 2, num_layers=3,
                                bidirectional=True, dropout=lstm_drop, batch_first=True)
        elif use_glove or use_pos or use_char_lstm:
            self.reduce_dimension = nn.Linear(lstm_input_size, config.hidden_size)
        self.aux_loss = aux_loss

        # Deformable的transformers
        self.encoder = BertModel(config)

        self.left_detector = MaskAwareSSNFuse(config)
        self.right_detector = MaskAwareSSNFuse(config)
        self.classifier = MaskAwareClassifier(config, n_classes)

        self.fpn_type = fpn_type
        self.use_msf = use_msf
        self.use_topk_query = use_topk_query
        self.__position_embedding = torch.nn.Parameter(
            make_position_embedding(
                self.config.max_position_embeddings, self.config.hidden_size)
        )
        if not use_topk_query:
            # 如果不是two stage，那么类似DETR，使用可学习的query
            self.query_embed = nn.Embedding(n_queries, config.hidden_size * 2)

        if fpn_pos_type is None:
            position_embedding = None
        elif fpn_pos_type == "bert":
            position_embedding = self.encoder.embeddings.position_embeddings.weight
        elif fpn_pos_type == "sinu":
            position_embedding = self.__position_embedding
        else:
            raise ValueError("Invalid fpn_pos_type {}".format(fpn_pos_type))
        if fpn_type == "uni":
            self.feature_net = PyramidFeatureNet(config, fpn_layer, fpn_drop, position_embedding)
        elif fpn_type == "bi":
            self.feature_net = BiPyramidFeatureNet(config, fpn_layer, fpn_drop, position_embedding)
        elif fpn_type == "uni-lstm":
            self.feature_net = PyramidFeatureLSTMNet(config, fpn_layer, fpn_drop)
        elif fpn_type == "bi-lstm":
            self.feature_net = BiPyramidFeatureLSTMNet(config, fpn_layer, fpn_drop)
        else:
            raise NotImplementedError("Fpn type \"{}\" is not implemented".format(fpn_type))

        self.feature_classifier = MaskAwareClassifier(config, n_classes)
        self.padding_position = nn.Embedding(1, self.feature_net.output_feature_size)
        self.feat_trans = nn.Linear(self.feature_net.output_feature_size, config.hidden_size * 2)
        self.pos_feat_norm = nn.LayerNorm(config.hidden_size)
        self.cls_feat_norm = nn.LayerNorm(config.hidden_size)

        if dec_type != "ssn":
            raise NotImplementedError("Dec type \"{}\" is not implemented".format(dec_type))

        self.decoder = SSNTrfDecoder(
            config,
            SSNTrfDecoderLayer(
                d_model=config.hidden_size,
                d_ffn=dec_intermediate_size,
                activation="relu",
                n_heads=dec_num_attention_heads,
            ),
            dec_layers
        )

        self.freeze_transformer = freeze_transformer
        self.split_epoch = split_epoch
        self.has_changed = False

        # self.init_weights()
        self.custom_init_weights()

    def freeze_bert_grad(self):
        if self.freeze_transformer or self.split_epoch > 0:
            print("Freeze transformer weights")
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def custom_init_weights(self):
        for child in self.children():
            if isinstance(child, MaskAwareClassifier):
                prior_prob = 0.01
                child.classifier.bias.data = torch.ones(self.n_classes) * -math.log((1 - prior_prob) / prior_prob)
            elif isinstance(child, BasePyramidFeatureNet):
                child.init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # if self.feat_trans is not None:
        #     for p in self.feat_trans.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform_(p)

    @staticmethod
    def combine(sub, sup_mask, pool_type="max"):
        sup = None
        if len(sub.shape) == len(sup_mask.shape):
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup == -1e30] = 0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup == -1e30] = 0
        return sup

    def gen_proposal_from_enc_output(self, encoding: torch.Tensor, padding_mask: torch.Tensor, sent_lens: torch.Tensor):
        """
        这里直接均匀sample出n_query个点。

        :param encoding: (bsz, len_seq, C)
        :param padding_mask: (bsz, len_seq)
        :param sent_lens: (bsz)
        :return: mem (bsz, n_queries, C), pos_ebd (bsz, n_queries, C)
        """
        (layer_feat, layer_mask), (feat, feat_mask), (left_boundary, right_boundary) = self.feature_net.forward(
            encoding, padding_mask, sent_lens
        )
        bsz, candidates, _ = feat.size()
        bsz, len_seq, _ = encoding.size()

        feat = torch.masked_scatter(feat, feat_mask.unsqueeze(-1),
                                    source=self.padding_position.weight.view(1, 1, -1).expand(bsz, feat.size(1), -1))

        feat_trans_out = self.feat_trans.forward(feat)
        position_feat, cls_feat = torch.split(feat_trans_out, self.config.hidden_size, dim=2)
        position_feat = self.pos_feat_norm.forward(position_feat)
        cls_feat = self.cls_feat_norm.forward(cls_feat)

        # Boundary
        left_boundary = torch.masked_fill(left_boundary, feat_mask, value=len_seq - 1)
        right_boundary = torch.masked_fill(right_boundary, feat_mask, value=0)

        proposal_left = torch.zeros((bsz, candidates, len_seq), dtype=feat.dtype, device=feat.device)
        proposal_left = torch.scatter(proposal_left, dim=-1, index=left_boundary.unsqueeze(-1), value=1.)
        proposal_right = torch.zeros((bsz, candidates, len_seq), dtype=feat.dtype, device=feat.device)
        proposal_right = torch.scatter(proposal_right, dim=-1, index=right_boundary.unsqueeze(-1), value=1.)

        # Classification
        proposal_cls = self.feature_classifier.forward(
            # feat, layer_feature[0], proposal_left, proposal_right, padding_mask, feat_mask
            cls_feat, feat_mask
        )

        scores = torch.sum(torch.softmax(proposal_cls, dim=-1)[..., 1:], dim=-1)
        # 要保证mask的排名靠后，并且mask部分的边界是不合法的边界，无法在hungarian匹配的时候产生匹配
        scores = torch.masked_fill(scores, feat_mask, -1)

        topk_scores, topk_indexes = torch.topk(scores, min(self.n_queries, candidates),
                                               dim=1, largest=True, sorted=False)
        topk_masks = topk_scores < 0
        query_pos = torch.gather(position_feat, 1,
                                 topk_indexes.unsqueeze(-1).expand(-1, -1, self.config.hidden_size))
        tgt = torch.gather(cls_feat, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, self.config.hidden_size))

        return (layer_feat, layer_mask), (feat, feat_mask), (query_pos, tgt, topk_masks), \
               (proposal_cls, proposal_left.detach(), proposal_right.detach()), topk_indexes

    def _common_forward(
        self, encodings: torch.tensor, context_masks: torch.tensor, seg_encoding: torch.tensor,
        context2token_masks: torch.tensor, token_masks: torch.tensor, pos_encoding: torch.tensor = None,
        wordvec_encoding: torch.tensor = None, char_encoding: torch.tensor = None,
        token_masks_char=None, char_count: torch.tensor = None,
        piggyback=None,
    ):
        bsz, _ = encodings.size()
        h_token = self.encoder.forward(input_ids=encodings, attention_mask=context_masks.float())
        if isinstance(h_token, BaseModelOutputWithPoolingAndCrossAttentions):
            h_token = h_token.last_hidden_state
        token_count = token_masks.long().sum(-1, keepdim=True)
        padding_mask = ~token_masks
        # 到这里embed变成token序列而不是BPE序列
        h_token = self.combine(h_token, context2token_masks, self.pool_type)
        embeds = [h_token]

        if self.use_pos:
            pos_embed = self.pos_embedding(pos_encoding)
            pos_embed = self.dropout(pos_embed)
            embeds.append(pos_embed)
        if self.use_glove:
            word_embed = self.word2vec_ebd(wordvec_encoding)
            word_embed = self.dropout(word_embed)
            embeds.append(word_embed)
        if self.use_char_lstm:
            char_count = char_count.view(-1)
            token_masks_char = token_masks_char
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count * bsz, max_char_count)

            char_encoding[char_count == 0][:, 0] = 101
            char_count[char_count == 0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input=char_embed, lengths=char_count.tolist(),
                                                                  enforce_sorted=False, batch_first=True)
            char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(bsz, max_token_count, max_char_count, self.char_size * 2)
            h_token_char = self.combine(char_embed, token_masks_char, "mean")
            embeds.append(h_token_char)

        h_token = torch.cat(embeds, dim=-1)

        if self.use_lstm:
            h_token = nn.utils.rnn.pack_padded_sequence(input=h_token, lengths=token_count.squeeze(-1).cpu().tolist(),
                                                        enforce_sorted=False, batch_first=True)
            h_token, (_, _) = self.lstm(h_token)
            h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)
        elif len(embeds) > 1:
            h_token = self.reduce_dimension(h_token)

        # Decoding
        bsz, len_tok, d_model = h_token.size()

        lay_feat, feat, topk_query, proposals, topk_indexes = self.gen_proposal_from_enc_output(
            h_token, padding_mask, token_count.cpu().squeeze(-1)
        )
        proposal_cls, proposal_left, proposal_right = proposals
        if self.use_msf:
            # 使用FPN特征
            dec_key, dec_key_mask = feat
            h_token = lay_feat[0][0]    # 用FPN的第一层作为预测边界的key
        else:
            # 使用token特征
            dec_key = h_token
            dec_key_mask = padding_mask
        if self.use_topk_query:
            # 用FPN的topk来初始化
            query_pos, tgt, tgt_masks = topk_query
            sa_tgt_mask = tgt_masks.unsqueeze(1).expand(-1, tgt_masks.size(-1), -1)
        else:
            # 用query embedding来初始化
            query_pos, tgt = torch.split(self.query_embed.weight, self.config.hidden_size, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bsz, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bsz, -1, -1)  # (bs, n_q, C)
            tgt_masks = None
            sa_tgt_mask = None

        output_classes = []
        output_lefts = []
        output_rights = []

        hs = self.decoder.forward(
            tgt, query_pos, dec_key, dec_key_mask, sa_q_mask=sa_tgt_mask, piggyback=piggyback,
        )
        # 处理输出
        for lvl, hidden_tgt in enumerate(hs):
            p_left = self.left_detector.forward(hidden_tgt, h_token, padding_mask, tgt_masks)
            p_right = self.right_detector.forward(hidden_tgt, h_token, padding_mask, tgt_masks)
            # p_cls = self.classifier.forward(hidden_tgt, h_token, p_left, p_right, padding_mask, tgt_masks)
            p_cls = self.classifier.forward(hidden_tgt, tgt_masks)
            output_classes.append(p_cls)
            output_lefts.append(p_left)
            output_rights.append(p_right)

        out = []
        if self.use_topk_query:
            # out.append({"box_aux_left_pred": fpn_aux_left_pred, "box_aux_right_pred": fpn_aux_right_pred,
            #             "box_aux_left_label": proposal_left, "box_aux_right_label": proposal_right})
            out.append({
                'entity_logits': proposal_cls,
                'p_left': proposal_left,
                'p_right': proposal_right,
                'tag': "BOUNDARY_MATCH",        # proposal只有边界匹配，这样和CE等价
                'max_gt_len': self.feature_net.num_layer,       # 最长只能propose这些长度，因此需要mask掉比这个长的gt
            })
        if self.aux_loss:
            for cls, p_left, p_right in zip(output_classes, output_lefts, output_rights):
                out.append({'entity_logits': cls, 'p_left': p_left, 'p_right': p_right})
        else:
            out.append({'entity_logits': output_classes[-1], 'p_left': output_lefts[-1], 'p_right': output_rights[-1]})

        if piggyback is not None:
            if "topk_indexes" in piggyback and topk_indexes is not None:
                piggyback["topk_indexes"] = topk_indexes.cpu()
            if "proposals" in piggyback and proposals is not None:
                piggyback["proposals"] = (proposal_cls.cpu(), proposal_left.cpu(), proposal_right.cpu())
        return out

    def _forward_train(self, epoch: int, *args, **kwargs):
        if not self.has_changed and epoch >= self.split_epoch and not self.freeze_transformer:
            tqdm.write("Now, update bert weights")
            for name, param in self.encoder.named_parameters():
                param.requires_grad = True
            self.has_changed = True
        # 这是Gumbel-softmax论文给的anneal方案
        # tau = max(0.5, math.exp(-2e-2 * epoch))
        return self._common_forward(*args, **kwargs)

    def _forward_eval(self, *args, **kwargs):
        return self._common_forward(*args, **kwargs)

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)
