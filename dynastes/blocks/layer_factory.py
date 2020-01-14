from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dynastes import layers as tfdl
from dynastes.probability.pseudoblocksparse_bijectors import BlockSparseStridedRoll1D


def get_1d_layer(type,
                 filters,
                 depth_multiplier,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 grouped=False,
                 group_size=1,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 kernel_normalizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
    if type.lower() == 'TimeDelayLayer1D'.lower():
        return tfdl.TimeDelayLayer1D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     dilation_rate=dilation_rate,
                                     padding=padding,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_normalizer=kernel_normalizer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     activity_regularizer=activity_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     bias_constraint=bias_constraint, **kwargs)
    elif type.lower() == 'DepthGroupwiseTimeDelayLayer1D'.lower():
        return tfdl.DepthGroupwiseTimeDelayLayer1D(depth_multiplier=depth_multiplier,
                                                   kernel_size=kernel_size,
                                                   strides=strides,
                                                   dilation_rate=dilation_rate,
                                                   padding=padding,
                                                   grouped=grouped,
                                                   group_size=group_size,
                                                   activation=activation,
                                                   use_bias=use_bias,
                                                   kernel_normalizer=kernel_normalizer,
                                                   kernel_initializer=kernel_initializer,
                                                   bias_initializer=bias_initializer,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                   kernel_constraint=kernel_constraint,
                                                   bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['Convolution1D'.lower(), 'Conv1D'.lower()]:
        return tfdl.DynastesConv1D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   dilation_rate=dilation_rate,
                                   padding=padding,
                                   activation=activation,
                                   use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   kernel_normalizer=kernel_normalizer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   activity_regularizer=activity_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['DepthwiseConv1D'.lower(), 'DepthwiseConvolution1D'.lower()]:
        return tfdl.DynastesDepthwiseConv1D(kernel_size=kernel_size,
                                            strides=strides,
                                            dilation_rate=dilation_rate,
                                            padding=padding,
                                            activation=activation,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            kernel_normalizer=kernel_normalizer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['Dense'.lower()]:
        return tfdl.DynastesDense(units=filters,
                                  activation=activation,
                                  use_bias=use_bias,
                                  kernel_initializer=kernel_initializer,
                                  kernel_normalizer=kernel_normalizer,
                                  bias_initializer=bias_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  activity_regularizer=activity_regularizer,
                                  kernel_constraint=kernel_constraint,
                                  bias_constraint=bias_constraint, **kwargs)


def get_1D_attention_layer(type,
                           strides,
                           dilation_rate,
                           num_heads,
                           padding,
                           multiquery_attention,
                           self_attention=True,
                           preshaped_q=True,
                           relative=False,
                           local=False,
                           sparse=False,
                           masked=False,
                           dropout_rate=0.,
                           max_relative_position=2,
                           blocksparse_bijector=BlockSparseStridedRoll1D,
                           lsh_bucket_length=4,
                           block_length=None,
                           filter_width=None,
                           mask_right=False,
                           add_relative_to_values=False,
                           heads_share_relative_embeddings=False):
    if type.lower() == 'LocalizedAttentionLayer1D'.lower():
        return tfdl.LocalizedAttentionLayer1D(strides=strides,
                                              dilation_rate=dilation_rate,
                                              num_heads=num_heads,
                                              padding=padding,
                                              preshaped_q=preshaped_q)
    elif type.lower() == 'Attention1D'.lower():
        return tfdl.Attention1D(num_heads=num_heads,
                                multiquery_attention=multiquery_attention,
                                self_attention=self_attention,
                                relative=relative,
                                masked=masked,
                                sparse=sparse,
                                local=local,
                                dropout_rate=dropout_rate,
                                max_relative_position=max_relative_position,
                                lsh_bucket_length=lsh_bucket_length,
                                block_length=block_length,
                                filter_width=filter_width,
                                mask_right=mask_right,
                                add_relative_to_values=add_relative_to_values,
                                heads_share_relative_embeddings=heads_share_relative_embeddings)
    elif type.lower() == 'PseudoBlockSparseAttention1D'.lower():
        return tfdl.PseudoBlockSparseAttention1D(num_heads=num_heads,
                                                 blocksparse_bijector=blocksparse_bijector(block_size=block_length),
                                                 multiquery_attention=multiquery_attention,
                                                 dropout_rate=dropout_rate,
                                                 block_size=block_length,
                                                 mask_right=mask_right)
