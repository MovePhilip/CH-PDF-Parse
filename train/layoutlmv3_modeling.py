# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LayoutLMv3 model."""

import collections
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.layoutlmv3.configuration_layoutlmv3 import LayoutLMv3Config
import wandb

logger = logging.get_logger("my_project")
#wandb.init(project="my_project")

# 在训练过程中或之后

_CONFIG_FOR_DOC = "LayoutLMv3Config"

LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    # See all LayoutLMv3 models at https://huggingface.co/models?filter=layoutlmv3
]

LAYOUTLMV3_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv3Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLMV3_MODEL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        bbox (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Batch of document images. Each image is divided into patches of shape `(num_channels, config.patch_size,
            config.patch_size)` and the total number of patches (=`patch_sequence_length`) equals to `((height /
            config.patch_size) * (width / config.patch_size))`.

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        bbox (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Batch of document images. Each image is divided into patches of shape `(num_channels, config.patch_size,
            config.patch_size)` and the total number of patches (=`patch_sequence_length`) equals to `((height /
            config.patch_size) * (width / config.patch_size))`.

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        labels (torch.LongTensor of shape (batch_size, sequence_length), optional) — Labels for computing the token classification loss.
            Indices should be in [0, ..., config.num_labels - 1].
"""


class LayoutLMv3PatchEmbeddings(nn.Module):
    """LayoutLMv3 image (patch) embeddings. This class also automatically interpolates the position embeddings for varying
    image sizes."""

    def __init__(self, config):
        super().__init__()

        image_size = (
            config.input_size
            if isinstance(config.input_size, collections.abc.Iterable)
            else (config.input_size, config.input_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, position_embedding=None):
        embeddings = self.proj(pixel_values)

        if position_embedding is not None:
            # interpolate the position embedding to the corresponding size
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1)
            position_embedding = position_embedding.permute(0, 3, 1, 2)
            patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
            position_embedding = F.interpolate(position_embedding, size=(patch_height, patch_width), mode="bicubic")
            embeddings = embeddings + position_embedding

        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class LayoutLMv3TextEmbeddings(nn.Module):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        
        if config.use_page_position_embeddings==True:
            self.use_page_position_embeddings=True
            self.page_position_embeddings = nn.Embedding(config.max_page_embedding, config.hidden_size)
        else:
            self.use_page_position_embeddings=False

        if config.use_reverse_position_embedding==True:
            self.use_reverse_position_embedding=True
            self.reverse_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        else:
            self.use_reverse_position_embedding=False

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        reverse_position_ids=None,
        page_position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        


        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings+=position_embeddings
        if self.use_reverse_position_embedding==True:
            reverse_position_embeddings = self.reverse_position_embeddings(reverse_position_ids)
            embeddings+=reverse_position_embeddings

        if  self.use_page_position_embeddings==True:
            try:
                page_position_embeddings=self.page_position_embeddings(page_position_ids)
            except:
                print(page_position_ids)
            embeddings += page_position_embeddings


        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        embeddings = embeddings + spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMv3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv3Config
    base_model_prefix = "layoutlmv3"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LayoutLMv3SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cogview_attention(self, attention_scores, alpha=32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original nn.Softmax(dim=-1)(attention_scores). Seems the new attention_probs
        will result in a slower speed and a little bias. Can use torch.allclose(standard_attention_probs,
        cogview_attention_probs, atol=1e-08) for comparison. The smaller atol (e.g., 1e-08), the better.
        """
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(dim=-1)(new_attention_scores)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The attention scores QT K/√d could be significantly larger than input elements, and result in overflow.
        # Changing the computational order into QT(K/√d) alleviates the problem. (https://arxiv.org/pdf/2105.13290.pdf)
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            attention_scores += (rel_pos + rel_2d_pos) / math.sqrt(self.attention_head_size)
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # Use the trick of the CogView paper to stablize training
        attention_probs = self.cogview_attention(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput
class LayoutLMv3SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Attention with LayoutLMv2->LayoutLMv3
class LayoutLMv3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv3SelfAttention(config)
        self.output = LayoutLMv3SelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Layer with LayoutLMv2->LayoutLMv3
class LayoutLMv3Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv3Attention(config)
        self.intermediate = LayoutLMv3Intermediate(config)
        self.output = LayoutLMv3Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LayoutLMv3Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        patch_height=None,
        patch_width=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaIntermediate
class LayoutLMv3Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput
class LayoutLMv3Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@add_start_docstrings(
    "The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV3_START_DOCSTRING,
)
class LayoutLMv3Model(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.text_embed:
            self.embeddings = LayoutLMv3TextEmbeddings(config)

        if config.visual_embed:
            # use the default pre-training parameters for fine-tuning (e.g., input_size)
            # when the input_size is larger in fine-tuning, we will interpolate the position embeddings in forward
            self.patch_embed = LayoutLMv3PatchEmbeddings(config)

            size = int(config.input_size / config.patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, config.hidden_size))
            self.pos_drop = nn.Dropout(p=0.0)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self.init_visual_bbox(image_size=(size, size))

            self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.encoder = LayoutLMv3Encoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc"
        )
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc"
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def calculate_visual_bbox(self, device, dtype, batch_size):
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, pixel_values):
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_MODEL_INPUTS_DOCSTRING.format("batch_size, token_sequence_length")
    )
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        reverse_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        page_position_ids=None
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")
        

        # print(input_ids.device)
        # print(pixel_values.device)
        #assert input_ids.device==pixel_values.device

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                reverse_position_ids=reverse_position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                page_position_ids=page_position_ids
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if pixel_values is not None:
            patch_height, patch_width = (
                int(pixel_values.shape[2] / self.config.patch_size),
                int(pixel_values.shape[3] / self.config.patch_size),
            )
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(device, dtype=torch.long, batch_size=batch_size)
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class LayoutLMv3ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.init_weights()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        reverse_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> word_labels = example["ner_tags"]

        >>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class LayoutLMv3ForTokenClassification_custom(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size=config.hidden_size
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # if config.num_labels < 10:
        self.texttype_classifier =nn.Sequential(
            nn.Dropout(config.text_type_classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.Tanh(),#nn.ELU(inplace=True)
            nn.Linear(config.hidden_size*2, 10, bias=False)
        )
        # else:
        #     self.texttype_classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)
        #定义二分类回归头
        self.use_binary_classification=config.use_binary_classification
        if config.use_binary_classification:
            self.binary_classfier = nn.Linear(2 * 768, 1)
            # Compute binary cross-entropy loss
            self.binary_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # 定义回归头
        self.regression_head =nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.Tanh(),
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1, bias=False)
        )
        self.newline_classifier =nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.Tanh(),
            nn.Linear(config.hidden_size*2, 2, bias=False)
        )

        self.token_classifier =nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.Tanh(),
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 5, bias=False)
        )
        self.class_weights = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0], requires_grad=False)
        self.loss_weights = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0], requires_grad=False)
        self.loss_dict={"first_loss":[],
                        "second_loss":[],
                        "third_loss":[],
                        "fourth_loss":[],
                        "fifth_loss":[]}
        self.batch_weight=False
        self.T=1
        self.init_weights()
    
    def check_loss_length(self):
        if len(self.loss_dict["first_loss"])<4:
            return 0
        if len(self.loss_dict["second_loss"])<4:
            return 0
        if len(self.loss_dict["third_loss"])<4:
            return 0
        if len(self.loss_dict["fifth_loss"])<4:
            return 
        self.batch_weight=True
        return 1
        

    def calculate_relative_weight(self,weight_exist):
         weightlist=[]
         if weight_exist[0]==1:
            w1=(self.loss_dict["first_loss"][-1]+self.loss_dict["first_loss"][-2])/(self.loss_dict["first_loss"][-3]+self.loss_dict["first_loss"][-4])
            weightlist.append(w1/self.T)
         if weight_exist[1]==1:
            w2=(self.loss_dict["second_loss"][-1]+self.loss_dict["second_loss"][-2])/(self.loss_dict["second_loss"][-3]+self.loss_dict["second_loss"][-4])
            weightlist.append(w2/self.T)
         if weight_exist[2]==1:
            w3=(self.loss_dict["third_loss"][-1]+self.loss_dict["third_loss"][-2])/(self.loss_dict["third_loss"][-3]+self.loss_dict["third_loss"][-4])
            weightlist.append(w3/self.T)
         if weight_exist[3]==1:
            weightlist.append(w3/self.T)
         if weight_exist[4]==1:
            w4=(self.loss_dict["fifth_loss"][-1]+self.loss_dict["fifth_loss"][-2])/(self.loss_dict["fifth_loss"][-3]+self.loss_dict["fifth_loss"][-4])
            weightlist.append(w4/self.T)
         w_i=torch.Tensor(weightlist)
         batch_weight = sum(weight_exist)*F.softmax(w_i, dim=-1)
         return batch_weight




    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        bbox: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1,4)
        attention_mask: Optional[torch.FloatTensor] = None, #(batch_size,sequence_length1)
        token_type_ids: Optional[torch.LongTensor] = None,#None
        position_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        reverse_position_ids: Optional[torch.LongTensor] = None,
        type_ids=None,       #(batch_size,sequence_length2)
        newline_id=None,     #(batch_size,sequence_length3)
        sentence_start=None,  #(batch_size,sequence_length2)
        order_id_text=None,       #(batch_size,sequence_length3)
        order_mask_text=None,  #(batch_size,sequence_length2)
        order_start_text=None,   #(batch_size,sequence_length3)
        order_id_caption=None,       #(batch_size,sequence_length3)
        order_mask_caption=None,  #(batch_size,sequence_length2)
        order_start_caption=None,   #(batch_size,sequence_length3)
        newline_start=None, #(batch_size,sequence_length3)
        epoch=None,
        page_position_id=None, #(batch_size,sequence_length1)这个和intpu_ids一样长
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        citation_start=None,
        citation_id=None
        
    ) -> Union[Tuple, TokenClassifierOutput]:
        input_ids_device=input_ids.device
        
        if pixel_values is not None:
            if pixel_values.device!=input_ids_device:
                pixel_values=pixel_values.to(input_ids_device)


        if type_ids is None and order_id_text is None and newline_id is None:
            if self.use_binary_classification:
                segment_type,newline,order_logits,token_type=self.predict_binary(input_ids=input_ids,
                                bbox= bbox,
                                attention_mask= attention_mask,
                                position_ids= position_ids,
                                reverse_position_ids=reverse_position_ids,
                                page_position_id=page_position_id,
                                sentence_start=sentence_start
                                )

            else:
                segment_type,newline,order_logits,token_type=self.predict(input_ids=input_ids,
                                bbox= bbox,
                                attention_mask= attention_mask,
                                position_ids= position_ids,
                                reverse_position_ids=reverse_position_ids,
                                page_position_id=page_position_id,
                                sentence_start=sentence_start,
                                pixel_values=pixel_values
                                )

            return segment_type,newline,order_logits,token_type
            



        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            reverse_position_ids=reverse_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            page_position_ids=page_position_id
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]


        #sequence_output = self.dropout(sequence_output)

        loss=0
        first_loss=None
        # 使用 torch.gather 选择每个segment的第一个token元素
        selected = torch.gather(sequence_output, 1, sentence_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        #print(f"selected.size(){selected.size()}")

        #first classifier,four type
        logits1 = self.texttype_classifier(selected)
        #print(f"logits1.shape:{logits1.size()}")
        if type_ids is not None:
            first_loss=F.cross_entropy(logits1.view(-1, logits1.size(-1)),type_ids.view(-1),ignore_index=-100,reduction='mean')
            #print(f"first_loss{first_loss}")
            # if torch.isnan(first_loss).any():
            #      print(f"input_ids:{input_ids}")
            #      print(f"logits1:{logits1}")
            #      print(f"selected:{selected}")
            #      print("something wrong")
        
        
        if newline_start is not None:
            selected2 = torch.gather(sequence_output, 1, newline_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        if order_start_text is not None:
            selected3 = torch.gather(sequence_output, 1, order_start_text.unsqueeze(2).expand(-1, -1, self.hidden_size))
        if order_start_caption is not None:
            selected4 = torch.gather(sequence_output, 1, order_start_caption.unsqueeze(2).expand(-1, -1, self.hidden_size))
        if citation_start is not None:
            selected5 = torch.gather(sequence_output, 1, citation_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        
        second_loss=None
        logits2=None
        #print(f"logits2.shape:{logits2.size()}")
        if type_ids is not None and newline_id[0,0]>-1:
            #second classifier,two type
            logits2 = self.newline_classifier(selected2)
            second_loss = F.cross_entropy(logits2.view(-1, logits2.size(-1)), newline_id.view(-1),ignore_index=-100,reduction='mean')# reduction='none',
            #print(f"second_loss{second_loss}")
            # if torch.isnan(second_loss).any():
            #     print(logits2)
        

        #计算排序损失
        third_loss=None
        logits3=None
        if self.use_binary_classification:
            if order_id_text is not None and order_mask_text.shape[1]>1 and order_mask_text[0,1]>0:
                third_loss=self.binary_loss(selected3,order_id_text,order_mask_text)
                #print(f"third_loss{third_loss}")
        else:
            if order_id_text is not None and len(order_mask_text[0])>1:
                if order_mask_text.shape[0]==1 and order_mask_text[0,1]>0:
                    logits3=self.regression_head(selected3)
                    if torch.max(logits3)>64:
                        penalize=True
                    else:
                        penalize=False
                    third_loss=self.list_mle(logits3.squeeze(-1),order_id_text,mask=order_mask_text,penalize=penalize)
                if order_mask_text.shape[0]>1 and order_mask_text[0,1]+order_mask_text[1,1]>0:
                    logits3=self.regression_head(selected3)
                    if torch.max(logits3)>64:
                        penalize=True
                    else:
                        penalize=False
                    if order_mask_text[0,1]==0:
                        logits3=logits3[1].unsqueeze(0)
                        order_id_text=order_id_text[1].unsqueeze(0)
                        order_mask_text=order_mask_text[1].unsqueeze(0)
                    elif order_mask_text[1,1]==0:
                        logits3=logits3[0].unsqueeze(0)
                        order_id_text=order_id_text[0].unsqueeze(0)
                        order_mask_text=order_mask_text[0].unsqueeze(0)
                    #print(f"logits3.shape:{logits3.size()}")
                    third_loss=self.list_mle(logits3.squeeze(-1),order_id_text,mask=order_mask_text,penalize=penalize)
                # if torch.isnan(third_loss).any():
                #     print(logits3)
        
        fourth_loss=None
        logits4=None
        if self.use_binary_classification:
            if order_id_caption is not None and order_mask_caption.shape[1]>1 and order_mask_caption[0,1]>0:
                fourth_loss=self.binary_loss(selected4,order_id_caption,order_mask_caption)
                #print(f"fourth_loss{fourth_loss}")
        else:
            if order_id_caption is not None and len(order_mask_caption[0])>1:
                if order_mask_caption.shape[0]==1 and order_mask_caption[0,1]>0:
                    logits4=self.regression_head(selected4)
                    penalize=False
                    fourth_loss=self.list_mle(logits4.squeeze(-1),order_id_caption,mask=order_mask_caption,penalize=penalize)
                    

                    if torch.isnan(fourth_loss).any():
                        logger.info(f"order_id_caption:{order_id_caption}")
                        logger.info(f"logits4:{logits4}")
                        logger.info(f"order_mask_caption:{order_mask_caption}")
                        print(logits4)
                
                if order_mask_caption.shape[0]>1 and order_mask_caption[0,1]+order_mask_caption[1,1]>0:
                    logits4=self.regression_head(selected4)
                    penalize=False

                    if order_mask_caption[0,1]==0:
                        logits4=logits4[1].unsqueeze(0)
                        order_id_caption=order_id_caption[1].unsqueeze(0)
                        order_mask_caption=order_mask_caption[1].unsqueeze(0)
                    
                    elif order_mask_caption[1,1]==0:
                        logits4=logits4[0].unsqueeze(0)
                        order_id_caption=order_id_caption[0].unsqueeze(0)
                        order_mask_caption=order_mask_caption[0].unsqueeze(0)

                    #print(f"logits3.shape:{logits3.size()}")
                    fourth_loss=self.list_mle(logits4.squeeze(-1),order_id_caption,mask=order_mask_caption,penalize=penalize)


                # if fourth_loss.item()<0:
                #     logger.info(f"order_id_caption:{order_id_caption}")
                #     logger.info(f"logits4:{logits4}")
                #     logger.info(f"order_mask_caption:{order_mask_caption}")
                    if torch.isnan(fourth_loss).any():
                        logger.info(f"order_id_caption:{order_id_caption}")
                        logger.info(f"logits4:{logits4}")
                        logger.info(f"order_mask_caption:{order_mask_caption}")
                        print(logits4)

        fifth_loss=None
        logits5=None
        #print(f"logits2.shape:{logits2.size()}")
        if citation_id is not None and citation_id[0,1]>-1:
            #second classifier,two type
            logits5 = self.token_classifier(selected5)
            class_weights = self.class_weights.to(logits5.device)
            fifth_loss = F.cross_entropy(logits5.view(-1, logits5.size(-1)), citation_id.view(-1),ignore_index=-100,weight=class_weights,reduction='mean')# reduction='none',
            #print(f"second_loss{second_loss}")
            # if torch.isnan(second_loss).any():
            #     print(logits2)

        #"first_loss":-1,"second_loss":-1,"third_loss":-1,"fourth_loss":-1,"fifth_loss":-1
        loss_dict={}


        if first_loss is not None:
            #loss=loss+first_loss#/first_loss.item()
            loss_dict["first_loss"]=first_loss#.item()

        if second_loss is not None:
                #loss=loss+second_loss#/second_loss.item()
                loss_dict["second_loss"]=second_loss#.item()
        if third_loss is not None:
                #loss=loss+6*third_loss#/third_loss.item()
                loss_dict["third_loss"]=third_loss#.item()
        if fourth_loss is not None:
                #loss=loss+fourth_loss#/fourth_loss.item()
                loss_dict["fourth_loss"]=fourth_loss#.item()
        if fifth_loss is not None:
                #loss=loss+fifth_loss#/fifth_loss.item()
                loss_dict["fifth_loss"]=fifth_loss#.item()

        #logger.info(str(loss_dict))
        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #if not return_dict:
        #wandb.log({"separate_loss": loss_dict})


        return loss_dict


        
        # if loss is not None:
        #     return loss
        # else:
        #     return (logits1,logits2,logits3) #+ outputs[1:]
            #return ((loss,) + output) if loss is not None else output

        # return TokenClassifierOutput(
        #     loss=loss,
        #     #logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


    def forward2(
        self,
        input_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        bbox: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1,4)
        attention_mask: Optional[torch.FloatTensor] = None, #(batch_size,sequence_length1)
        token_type_ids: Optional[torch.LongTensor] = None,#None
        position_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        reverse_position_ids: Optional[torch.LongTensor] = None,
        type_ids=None,       #(batch_size,sequence_length2)
        newline_id=None,     #(batch_size,sequence_length3)
        sentence_start=None,  #(batch_size,sequence_length2)
        order_id_text=None,       #(batch_size,sequence_length3)
        order_mask_text=None,  #(batch_size,sequence_length2)
        order_start_text=None,   #(batch_size,sequence_length3)
        order_id_caption=None,       #(batch_size,sequence_length3)
        order_mask_caption=None,  #(batch_size,sequence_length2)
        order_start_caption=None,   #(batch_size,sequence_length3)
        newline_start=None, #(batch_size,sequence_length3)
        epoch=None,
        page_position_id=None, #(batch_size,sequence_length1)这个和intpu_ids一样长
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        citation_start=None,
        citation_id=None
        
    ) -> Union[Tuple, TokenClassifierOutput]:
        input_ids_device=input_ids.device
        
        if pixel_values is not None:
            if pixel_values.device!=input_ids_device:
                pixel_values=pixel_values.to(input_ids_device)


        if type_ids is None and order_id_text is None and newline_id is None:
            if self.use_binary_classification:
                segment_type,newline,order_logits,token_type=self.predict_binary(input_ids=input_ids,
                                bbox= bbox,
                                attention_mask= attention_mask,
                                position_ids= position_ids,
                                reverse_position_ids=reverse_position_ids,
                                page_position_id=page_position_id,
                                sentence_start=sentence_start
                                )

            else:
                segment_type,newline,order_logits,token_type=self.predict(input_ids=input_ids,
                                bbox= bbox,
                                attention_mask= attention_mask,
                                position_ids= position_ids,
                                reverse_position_ids=reverse_position_ids,
                                page_position_id=page_position_id,
                                sentence_start=sentence_start,
                                pixel_values=pixel_values
                                )

            return segment_type,newline,order_logits,token_type
            



        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            reverse_position_ids=reverse_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            page_position_ids=page_position_id
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]


        #sequence_output = self.dropout(sequence_output)

        loss=0
        first_loss=None
        # 使用 torch.gather 选择每个segment的第一个token元素
        selected = torch.gather(sequence_output, 1, sentence_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        #print(f"selected.size(){selected.size()}")

        #first classifier,four type
        logits1 = self.texttype_classifier(selected)
        #print(f"logits1.shape:{logits1.size()}")
        if type_ids is not None:
            first_loss=F.cross_entropy(logits1.view(-1, logits1.size(-1)),type_ids.view(-1),ignore_index=-100,reduction='mean')
            #print(f"first_loss{first_loss}")
            # if torch.isnan(first_loss).any():
            #      print(f"input_ids:{input_ids}")
            #      print(f"logits1:{logits1}")
            #      print(f"selected:{selected}")
            #      print("something wrong")
        
        
        if newline_start is not None:
            selected2 = torch.gather(sequence_output, 1, newline_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        if order_start_text is not None:
            selected3 = torch.gather(sequence_output, 1, order_start_text.unsqueeze(2).expand(-1, -1, self.hidden_size))
        if order_start_caption is not None:
            selected4 = torch.gather(sequence_output, 1, order_start_caption.unsqueeze(2).expand(-1, -1, self.hidden_size))
        if citation_start is not None:
            selected5 = torch.gather(sequence_output, 1, citation_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        
        second_loss=None
        logits2=None
        #print(f"logits2.shape:{logits2.size()}")
        if type_ids is not None and newline_id[0,0]>-1:
            #second classifier,two type
            logits2 = self.newline_classifier(selected2)
            second_loss = F.cross_entropy(logits2.view(-1, logits2.size(-1)), newline_id.view(-1),ignore_index=-100,reduction='mean')# reduction='none',
            #print(f"second_loss{second_loss}")
            # if torch.isnan(second_loss).any():
            #     print(logits2)
        

        #计算排序损失
        third_loss=None
        logits3=None
        if self.use_binary_classification:
            if order_id_text is not None and order_mask_text.shape[1]>1 and order_mask_text[0,1]>0:
                third_loss=self.binary_loss(selected3,order_id_text,order_mask_text)
                #print(f"third_loss{third_loss}")
        else:
            if order_id_text is not None and len(order_mask_text[0])>1 and order_mask_text[0,1]>0:
                #我觉得在这个前面加上个layer_normalization可能是很有必要的
                logits3=self.regression_head(selected3)
                if torch.max(logits3)>64:
                    penalize=True
                else:
                    penalize=False
                #print(f"logits3.shape:{logits3.size()}")
                third_loss=self.list_mle(logits3.squeeze(-1),order_id_text,mask=order_mask_text,penalize=penalize)
                if torch.isnan(third_loss).any():
                    print(logits3)
        
        fourth_loss=None
        logits4=None
        if self.use_binary_classification:
            if order_id_caption is not None and order_mask_caption.shape[1]>1 and order_mask_caption[0,1]>0:
                fourth_loss=self.binary_loss(selected4,order_id_caption,order_mask_caption)
                #print(f"fourth_loss{fourth_loss}")
        else:
            if order_id_caption is not None and len(order_mask_caption[0])>1 and order_mask_caption[0,1]>0:
                logits4=self.regression_head(selected4)
                penalize=False
                #print(f"logits3.shape:{logits3.size()}")
                fourth_loss=self.list_mle(logits4.squeeze(-1),order_id_caption,mask=order_mask_caption,penalize=penalize)
                if fourth_loss.item()<0:
                    logger.info(f"order_id_caption:{order_id_caption}")
                    logger.info(f"logits4:{logits4}")
                    logger.info(f"order_mask_caption:{order_mask_caption}")
                if torch.isnan(fourth_loss).any():
                    print(logits4)

        fifth_loss=None
        logits5=None
        #print(f"logits2.shape:{logits2.size()}")
        if citation_id is not None and citation_id[0,1]>-1:
            #second classifier,two type
            logits5 = self.token_classifier(selected5)
            class_weights = self.class_weights.to(logits5.device)
            fifth_loss = F.cross_entropy(logits5.view(-1, logits5.size(-1)), citation_id.view(-1),ignore_index=-100,weight=class_weights,reduction='mean')# reduction='none',
            #print(f"second_loss{second_loss}")
            # if torch.isnan(second_loss).any():
            #     print(logits2)

        #"first_loss":-1,"second_loss":-1,"third_loss":-1,"fourth_loss":-1,"fifth_loss":-1
        if self.batch_weight==False:
            self.check_loss_length()
        
        weight_exist=[0,0,0,0,0]
        if first_loss is not None:
            weight_exist[0]=1
            if self.batch_weight==True:
                self.loss_dict["first_loss"].pop()


        if second_loss is not None:
            weight_exist[0]=1
            if self.batch_weight==True:
                self.loss_dict["first_loss"].pop()

        if third_loss is not None:
            weight_exist[0]=1
            if self.batch_weight==True:
                self.loss_dict["first_loss"].pop()

        if fourth_loss is not None:
            weight_exist[0]=1
            if self.batch_weight==True:
                self.loss_dict["first_loss"].pop()

        if fifth_loss is not None:
            weight_exist[0]=1
            if self.batch_weight==True:
                self.loss_dict["first_loss"].pop()
        if self.batch_weight==True:
            batch_weight=self.calculate_relative_weight(weight_exist=weight_exist)


        #logger.info(str(self.loss_dict))
        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #if not return_dict:
        #wandb.log({"separate_loss": {loss_dict})
        if loss is not None:
            return loss
        else:
            return (logits1,logits2,logits3) #+ outputs[1:]


    def predict(self,input_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        bbox: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1,4)
        attention_mask: Optional[torch.FloatTensor] = None, #(batch_size,sequence_length1)
        token_type_ids: Optional[torch.LongTensor] = None,#None
        position_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        reverse_position_ids: Optional[torch.LongTensor] = None,
        page_position_id=None, #(batch_size,sequence_length1)这个和intpu_ids一样长
        sentence_start=None,  #(batch_size,sequence_length2)
        pixel_values:Optional[torch.LongTensor] = None
        ):
        outputs = self.layoutlmv3(
                    input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    reverse_position_ids=reverse_position_ids,
                    page_position_ids=page_position_id,
                    pixel_values=pixel_values
                )
        if input_ids is not None:
            input_shape = input_ids.size()

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        # 使用 torch.gather 选择每个segment的第一个token元素
        selected = torch.gather(sequence_output, 1, sentence_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        #first classifier,four type
        type_logits = self.texttype_classifier(selected)#预测是什么类型的文本
        newline_logits = self.newline_classifier(selected)#预测是否是新行，这里是把那个无关的行也做了预测，
        order_logits=self.regression_head(selected)#预测文本行顺序，这里也是把那个无关的行也做了预测
        token_type_logits = self.token_classifier(sequence_output)#预测那个token类型
        token_type=torch.argmax(token_type_logits,dim=-1)
        segment_type=torch.argmax(type_logits,dim=-1)
        newline=torch.argmax(newline_logits,dim=-1)
        return segment_type.cpu().numpy(),newline.cpu().numpy(),order_logits.cpu().detach().numpy(),token_type.cpu().detach().numpy()
    
    def predict_binary(self,input_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        bbox: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1,4)
        attention_mask: Optional[torch.FloatTensor] = None, #(batch_size,sequence_length1)
        token_type_ids: Optional[torch.LongTensor] = None,#None
        position_ids: Optional[torch.LongTensor] = None, #(batch_size,sequence_length1)
        reverse_position_ids: Optional[torch.LongTensor] = None,
        page_position_id=None, #(batch_size,sequence_length1)这个和intpu_ids一样长
        sentence_start=None,  #(batch_size,sequence_length2)
        ):
        outputs = self.layoutlmv3(
                    input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    reverse_position_ids=reverse_position_ids,
                    page_position_ids=page_position_id
                )
        if input_ids is not None:
            input_shape = input_ids.size()

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        # 使用 torch.gather 选择每个segment的第一个token元素
        selected = torch.gather(sequence_output, 1, sentence_start.unsqueeze(2).expand(-1, -1, self.hidden_size))
        token_type_logits = self.token_classifier(sequence_output)#预测那个token类型
        token_type=torch.argmax(token_type_logits,dim=-1)


        #first classifier,four type
        type_logits = self.texttype_classifier(selected)#预测是什么类型的文本
        newline_logits = self.newline_classifier(selected)#预测是否是新行，这里是把那个无关的行也做了预测，
        segment_type=torch.argmax(type_logits,dim=-1)
        newline=torch.argmax(newline_logits,dim=-1)

        batch_size, seq_len, hidden_size = selected.shape
        batch_indices = torch.arange(batch_size).unsqueeze(1)  # Shape [batch_size, 1]
        # Use upper triangular part indices
        i, j = torch.triu_indices(seq_len, seq_len, 1)  # Start from 1 to avoid diagonal  #row_indices, col_indices,

        # Select only the relevant parts
        expanded_out1 = selected[batch_indices, i, :]
        expanded_out2 = selected[batch_indices, j, :]
        # print(f"expanded_out1{expanded_out1.shape}")
        # print(f"expanded_out2{expanded_out2.shape}")



        x = torch.cat((expanded_out1, expanded_out2), dim=-1)
        # Flatten the concatenated tensor to apply linear layer
        #x = x.view(-1, 2 * hidden_size)
        scores = self.binary_classfier(x).squeeze(-1)

        # 创建全零矩阵
        full_matrix = torch.zeros(batch_size, seq_len, seq_len).to(input_ids.device)
        # print(f"scores{scores.shape}")
        # 填充上三角部分
        full_matrix[:, i, j] = scores

        # 计算下三角部分的值
        full_matrix[:, j, i] = 0.5 - full_matrix[:, i, j]#这里我不确定是否正确，待会训练好看下
        #print(full_matrix.shape)


        # Masking
        # Masking
        # mask_i = mask[:, i]  # [batch_size, num_pairs]
        # mask_j = mask[:, j]  # [batch_size, num_pairs]
        # # print(f"mask_i{mask_i.shape}")
        # valid_pairs = (mask_i & mask_j).float()  # Element-wise AND to ensure both are valid
        #print(f"valid_pairs{valid_pairs.shape}")

        return segment_type.cpu().numpy(),newline.cpu().numpy(),full_matrix.sum(dim=-1).cpu().detach().numpy(),token_type.cpu().detach().numpy()

                        


    def listnet_loss(self, predict, target, mask):
        """
        Modifies the ListNet loss function to account for padding tokens.
        
        Args:
        - predict (torch.Tensor): Predictions tensor of shape (batch_size, sequence_length).
        - target (torch.Tensor): Target tensor of shape (batch_size, sequence_length).
        - mask (torch.Tensor): Mask tensor of shape (batch_size, sequence_length) containing 1 for valid tokens and 0 for pads.
        
        Returns:
        - torch.Tensor: The calculated loss.
        """
        # Apply mask to target and prediction by setting pad positions to zero
        #predict = predict * mask
        
        transformed_mask = torch.where(mask == 1, torch.tensor(0, dtype=mask.dtype), torch.tensor(-999, dtype=mask.dtype))  
        target = target + transformed_mask
        
        # Compute softmax over the sequence dimension
        top1_target = F.softmax(target, dim=1)
        top1_predict = F.softmax(predict, dim=1)
        
        # Compute the loss only where mask is 1
        loss = -torch.sum(top1_target * torch.log(top1_predict + 1e-10), dim=1)  # add a small value to prevent log(0)
        #masked_loss = loss * mask[:, 0]  # Apply mask to loss if needed (mask should be the same across the sequence)

        # Calculate mean loss only over the masked/valid entries
        sum_loss =loss.sum()
        sum_mask = mask.sum()
        mean_loss = sum_loss / sum_mask

        return mean_loss

    def list_mle2(self, y_pred, y_true, mask, k=None,penalize=False):
        # y_pred : batch x n_items
        # y_true : batch x n_items 
        if k is not None:
            sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
            y_pred = y_pred[:, sublist_indices] 
            y_true = y_true[:, sublist_indices] 
    
        _, indices = y_true.sort(descending=True, dim=-1)
        
        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
        
        #同意缩放到[-30,30]刻度
        max_abs = torch.max(torch.abs(pred_sorted_by_true))#.detach()  # 使用 detach 来确保 max_abs 不需要梯度
        scale_factor = 30 / max_abs  # 计算缩放因子
        pred_sorted_by_true = pred_sorted_by_true * scale_factor  # 应用缩放因子

        if mask is not None:
            mask_sorted = mask.gather(dim=1, index=indices)
            small_number = 1e-4
            mask_transformed = torch.where(mask_sorted == 0, torch.full_like(mask_sorted, small_number), mask_sorted)
            
            #pred_sorted_by_true = pred_sorted_by_true * mask_transformed  # Apply mask

        # Compute probabilities
        exp_preds = pred_sorted_by_true.exp()
        if mask is not None:
            exp_preds = exp_preds * mask_transformed  # Zero out the ignored positions

        cumsums = exp_preds.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        
        # Avoid division by zero
        valid_cumsums = cumsums + 1e-10
        listmle_loss = torch.log(valid_cumsums) - pred_sorted_by_true

        if mask is not None:
            listmle_loss = listmle_loss * mask_sorted  # Apply mask to loss
        
        listmle_loss=listmle_loss.sum(dim=1)/mask.sum(dim=1)
        # Compute final loss
        # if penalize==True:
            
        return listmle_loss.mean()#+1e-4*torch.sum(torch.pow(pred_sorted_by_true,2))/mask.sum()#加上一个正则化项
        # else:
        #     return listmle_loss.mean()

    def list_mle(self, scores, targets, mask,penalize=False):
        """
        计算 ListMLE 损失，支持 mask 以指定需要排序的项目。

        参数:
            scores: Tensor of shape (batch_size, list_size)
                模型预测的分数。
            targets: Tensor of shape (batch_size, list_size)
                真实的相关性标签。通常较高的值表示更高的相关性。
            mask: Tensor of shape (batch_size, list_size)
                指定需要排序的项目。值为1的地方需要排序，值为0的地方不需要排序。
        
        返回:
            loss: Scalar tensor representing the ListMLE 损失。
        """
        # 确保 scores, targets 和 mask 的形状一致
        assert scores.size() == targets.size() == mask.size(), "scores, targets, and mask must have the same shape"

        batch_size, list_size = scores.size()

        # 将 mask 应用于 targets，将 mask=0 的项目的目标值设为 -inf，确保它们在排序后位于末尾
        # 这样它们不会影响排序和损失计算
        masked_targets = targets.clone()
        masked_targets = masked_targets.masked_fill(mask == 0, -float('inf'))

        # 根据 masked_targets 对每个列表中的项目进行排序，降序排列
        # sorted_targets: (batch_size, list_size)
        # indices: (batch_size, list_size)
        sorted_targets, indices = torch.sort(masked_targets, dim=1, descending=True)

        # 根据排序后的 indices 获取对应的 scores
        sorted_scores = torch.gather(scores, dim=1, index=indices)  # (batch_size, list_size)

        raw_max=torch.max(sorted_scores, dim=1).values
        sorted_scores = sorted_scores-raw_max.unsqueeze(1)


        # 获取排序后的 mask
        sorted_mask = torch.gather(mask, dim=1, index=indices)  # (batch_size, list_size)

        # 对于 mask=0 的项目，将它们的 scores 设为一个很小的值，以确保在计算 logsumexp 时被忽略
        # 这里使用 -1e9 作为一个足够小的数
        sorted_scores = sorted_scores.masked_fill(sorted_mask == 0, -1e9)

        # 对排序后的 scores 进行反转，以便从后向前计算累积的 logsumexp
        reversed_scores = torch.flip(sorted_scores, dims=[1])  # (batch_size, list_size)

        # 计算累积的 logsumexp
        cum_logsumexp_reversed = torch.logcumsumexp(reversed_scores, dim=1)  # (batch_size, list_size)

        # 再次反转回来，以匹配原始排序
        cum_logsumexp = torch.flip(cum_logsumexp_reversed, dims=[1])  # (batch_size, list_size)

        # 计算 sum_i s_i，其中仅包括 mask=1 的项目
        sum_scores = torch.sum(sorted_scores.masked_fill(sorted_mask == 0, 0), dim=1)  # (batch_size)

        # 计算 sum_i logsumexp_{k>=i} s_k，其中仅包括 mask=1 的项目
        sum_logsumexp = torch.sum(cum_logsumexp.masked_fill(sorted_mask == 0, 0), dim=1)  # (batch_size)

        # 损失为 -sum_scores + sum_logsumexp
        loss = -sum_scores + sum_logsumexp  # (batch_size)

        # 最终返回批量的平均损失
        return torch.mean(loss)




    def binary_loss(self,encoder_outputs, labels,mask):
        batch_size, seq_len, hidden_size = encoder_outputs.shape
        batch_indices = torch.arange(batch_size).unsqueeze(1)  # Shape [batch_size, 1]
        # Use upper triangular part indices
        i, j = torch.triu_indices(seq_len, seq_len, 1)  # Start from 1 to avoid diagonal

        # Select only the relevant parts
        expanded_out1 = encoder_outputs[batch_indices, i, :]
        expanded_out2 = encoder_outputs[batch_indices, j, :]
        # print(f"expanded_out1{expanded_out1.shape}")
        # print(f"expanded_out2{expanded_out2.shape}")



        x = torch.cat((expanded_out1, expanded_out2), dim=-1)
        # Flatten the concatenated tensor to apply linear layer
        #x = x.view(-1, 2 * hidden_size)
        scores = self.binary_classfier(x).squeeze(-1)
        # print(f"scores{scores.shape}")



        # Masking
        # Masking
        mask_i = mask[:, i]  # [batch_size, num_pairs]
        mask_j = mask[:, j]  # [batch_size, num_pairs]
        # print(f"mask_i{mask_i.shape}")
        valid_pairs = (mask_i & mask_j).float()  # Element-wise AND to ensure both are valid
        #print(f"valid_pairs{valid_pairs.shape}")



       
        # Create target tensor for upper triangular part
        true_labels = labels.unsqueeze(1) < labels.unsqueeze(2)
        target = true_labels[:, i, j].float()
        # print(f"target{target.shape}")

        # # Apply mask to scores and target
        # scores = scores * valid_pairs
        # target = target * valid_pairs

        loss = self.binary_criterion(scores, target)

        loss = (loss * valid_pairs).sum() / valid_pairs.sum()
        if torch.isnan(valid_pairs.sum()).any():
            print("error")
            print("error")

        #print(f"valid_pairs.sum(){valid_pairs.sum()}")
        return loss



