
"""Attention layer.

Original paper:
Attention is all you need. (https://arxiv.org/abs/1706.03762)

You can also use torch.nn.MultiheadAttention class.
https://pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html

ref) Sample code

https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py

https://github.com/halhorn/deep_dialog_tutorial/blob/master/deepdialog/transformer/attention.py

https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward method to return queried values.

        Input tensors' size should be `(batch, len, dim)` or
        `(batch, num, len, dim)`.

        Args:
            q (torch.Tensor): sample of size `(*, len_q, d_k)`.
            k (torch.Tensor): Key of size `(*, len_k, d_k)`.
            v (torch.Tensor): Value of size `(*, len_k, d_v)`.

        Returns:
            y (torch.Tensor): Queried value of size `(*, len_q, d_v)`.
        """

        d_k = k.size(2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)

        return y


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    In Attentive NP, variable names have the following meaning.

    * Query: x target
    * Keys: x context
    * Values: representations
    * Outputs: aggregated representation

    Args:
        k_dim (int): Dimension size of key and query.
        v_dim (int): Dimension size of value.
        d_k (int): Dimension size of key and query in attention module.
        d_v (int): Dimension size of value in attention module.
        n_head (int): Number of heads.
    """

    def __init__(self, k_dim: int, v_dim: int, d_k: int, d_v: int,
                 n_head: int) -> None:
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(k_dim, n_head * d_k, bias=False)
        self.w_k = nn.Linear(k_dim, n_head * d_k, bias=False)
        self.w_v = nn.Linear(v_dim, n_head * d_v, bias=False)
        self.fc_y = nn.Linear(n_head * d_v, v_dim, bias=False)

        self.attention = ScaledDotProductAttention()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Foward method.

        Args:
            q (torch.Tensor): Query of size `(batch, len_q, k_dim)`.
            k (torch.Tensor): Key of size `(batch, len_k, k_dim)`.
            v (torch.Tensor): Value of size `(batch, len_k, v_dim)`.

        Returns:
            y (torch.Tensor): Queried Values of size `(batch, len_q, v_dim)`.
        """

        if q.size(2) != k.size(2):
            raise ValueError("Dimension of queue and key should be same: "
                             f"queue size = {q.size()}, key size = {k.size()}")

        if k.size(1) != v.size(1):
            raise ValueError("Length of key and value should be same: "
                             f"key size = {k.size()}, value size = {v.size()}")

        size_b, len_q, _ = q.size()
        len_k = k.size(1)

        # Linear projection and split matrices to each head
        # (batch, len, dim) -> (batch, len, n_head, d)
        q = self.w_q(q).view(size_b, len_q, self.n_head, self.d_k)
        k = self.w_k(k).view(size_b, len_k, self.n_head, self.d_k)
        v = self.w_v(v).view(size_b, len_k, self.n_head, self.d_v)

        # Transpose for attention: (batch, n_head, len, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Pass through attention
        y = self.attention(q, k, v)

        # Combine each head: (batch, n_head, len, d) -> (batch, len, n*d)
        y = y.transpose(1, 2).contiguous().view(size_b, len_q, -1)

        # Linear projection: (batch, len, n*d) -> (batch, len_q, v_dim)
        y = self.fc_y(y)

        return y


class SelfAttention(MultiHeadAttention):
    """Self Attention with multi head module.

    Args:
        x_dim (int): Dimension of input x.
        d_k (int): Dimension size of x in attention module.
        n_head (int): Number of heads.
    """

    def __init__(self, x_dim: int, d_x: int, n_head: int) -> None:
        super().__init__(x_dim, x_dim, d_x, d_x, n_head)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for self-attention.

        Args:
            x (torch.Tensor): Input of size `(batch, len_x, x_dim)`.

        Returns:
            y (torch.Tensor): Queried Values of size `(batch, len_x, x_dim)`.
        """

        return super().forward(x, x, x)
