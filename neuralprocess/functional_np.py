
"""Functional Neural Process.

C. Louizos et al., "The Functional Neural Process"
http://arxiv.org/abs/1906.08324

Reference)
https://github.com/AMLab-Amsterdam/FNP
"""

from typing import Tuple, Dict

from itertools import product
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import RelaxedBernoulli

from .base_np import BaseNP, nll_normal


def logitexp(logp: Tensor) -> Tensor:
    """Logp to logit with numerical stabiliztion.

    https://github.com/pytorch/pytorch/issues/4007

    Args:
        logp (torch.Tensor): Log probability.

    Returns:
        logit (torch.Tensor): Logit.
    """

    pos = torch.clamp(logp, min=-math.log(2))
    neg = torch.clamp(logp, max=-math.log(2))
    neg_val = neg - (1 - neg.exp()).log()
    pos_val = -(torch.clamp(torch.expm1(-pos), min=1e-20)).log()

    return pos_val + neg_val


class DAGEmbedding(nn.Module):
    """DAG embedding: p(G|U_R) p(A|U_R, U_M).

    Args:
        u_dim (int): Dimension size of latent u.
        temperature (float, optional): Temperature for relaxed bernoulli dist.
    """

    def __init__(self, u_dim: int, temperature: float = 0.3):
        super().__init__()

        self.temperature = temperature
        self.scale = nn.Parameter((u_dim * torch.ones(1)).sqrt())

    def forward(self, u_c: Tensor, u_t: Tensor) -> Tuple[Tensor]:
        """Forward method to return graph G, A.

        Args:
            u_c (torch.Tensor): u input for context, size `(b, n, u_dim)`.
            u_t (torch.Tensor): u input for target, size `(b, m, u_dim)`.

        Returns:
            graph (torch.Tensor): Sampled DAG `G`, size `(b, n, n)`.
            bipartite (torch.Tensor): Bipartite graph `A`, size `(b, m, n)`.
        """

        graph = self._sample_dag(u_c)
        bipartite = self._sample_bipartite(u_c, u_t)

        return graph, bipartite

    def _sample_dag(self, u_c: Tensor) -> Tensor:
        """Samples DAG from context data: p(G|U_R).

        Args:
            u_c (torch.Tensor): u input for context, size `(b, n, u_dim)`.

        Returns:
            graph (torch.Tensor): Sampled DAG, size `(b, n, n)`.
        """

        # Data size
        b, n, _ = u_c.size()

        # Ordering by log CDF
        log_cdf = (0.5 * (u_c / 2 ** 0.5).erf() + 0.5).log().sum(dim=-1)
        u_c_sorted, sort_idx = log_cdf.sort()

        # Indices of upper triangular adjacency matrix for DAG
        indices = torch.triu_indices(n, n, offset=1)

        # Latent pairs (b, num_pairs)
        pair_0 = u_c_sorted[:, indices[0]]
        pair_1 = u_c_sorted[:, indices[1]]

        # Compute logits for each pair
        logp = -0.5 * (pair_0 - pair_1) ** 2 / self.scale
        logits = logitexp(logp)

        # Sample graph from bernoulli dist (b, num_pairs)
        dist = RelaxedBernoulli(logits=logits, temperature=self.temperature)
        sorted_graph = dist.rsample()

        # Embed upper triangular to adjancency matrix
        graph = u_c.new_zeros((b, n, n))
        graph[:, indices[0], indices[1]] = sorted_graph

        # Unsort index of DAG to data order
        col_idx = torch.argsort(sort_idx)
        col_idx = col_idx.unsqueeze(1).repeat(1, n, 1)

        # Swap to unsort: 1. columns, 2. indices as columns
        graph = torch.gather(graph, -1, col_idx)
        graph = torch.gather(graph.permute(0, 2, 1), -1, col_idx)
        graph = graph.permute(0, 2, 1)

        return graph

    def _sample_bipartite(self, u_c: Tensor, u_t: Tensor) -> Tensor:
        """Samples bipartite: p(A|U_R, U_M).

        Args:
            u_c (torch.Tensor): u input for context, size `(b, n, u_dim)`.
            u_t (torch.Tensor): u input for target, size `(b, m, u_dim)`.

        Returns:
            bipartite (torch.Tensor): Bipartite graph, size `(b, m, n)`.
        """

        b, n, _ = u_c.size()
        m = u_t.size(1)
        indices = list(product(range(m), range(n)))

        # Latent pairs (b, num_pairs, u_dim)
        pair_0 = u_t[:, indices[0]]
        pair_1 = u_c[:, indices[1]]

        # Compute logits for each pair
        logp = -0.5 * ((pair_0 - pair_1) ** 2).sum(dim=-1) / self.scale
        logits = logitexp(logp)

        # Sample graph from bernoulli dist (b, num_pairs)
        dist = RelaxedBernoulli(logits=logits, temperature=self.temperature)
        p_edges = dist.rsample()

        # Embed values
        bipartite = u_c.new_zeros((b, m, n))
        bipartite[:, indices[0], indices[1]] = p_edges

        return bipartite


class FunctionalNP(BaseNP):
    """Functional Neural Process class.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        h_dim (int): Dimension size of h (hidden representation).
        u_dim (int): Dimension size of u (encoded inputs).
        z_dim (int): Dimension size of z (stochastic latent).
    """

    def __init__(self, x_dim: int, y_dim: int, h_dim: int, u_dim: int,
                 z_dim: int):
        super().__init__()

        # h = f(x): Input transformation
        self.f_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
        )

        # p(u|x)
        self.p_u = nn.Linear(h_dim, u_dim * 2)

        # p(G|U_R) p(A|U_R, U_M)
        self.p_ga = DAGEmbedding(u_dim)

        # q(z|x)
        self.q_z = nn.Linear(h_dim, z_dim * 2)

        # g(y): Linear embedding of labels
        self.g_y = nn.Linear(y_dim, z_dim * 2)

        # p(y|z)
        self.p_y = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim * 2),
        )

    def sample(self, x_context: Tensor, y_context: Tensor, x_target: Tensor
               ) -> Tuple[Tensor, Tensor]:
        """Samples queried y target.

        Args:
            x_context (torch.Tensor): x for context, size
                `(batch_size, num_context, x_dim)`.
            y_context (torch.Tensor): y for context, size
                `(batch_size, num_context, y_dim)`.
            x_target (torch.Tensor): x for target, size
                `(batch_size, num_target, x_dim)`.

        Returns:
            mu (torch.Tensor): y queried by target x and encoded
                representation, size `(batch_size, num_target, y_dim)`.
            var (torch.Tensor): Variance of y, size
                `(batch_size, num_target, y_dim)`.
        """

        # Representation
        h_c = self.f_x(x_context)
        h_t = self.f_x(x_target)

        # Sample u
        mu_u_c, logvar_u_c = torch.chunk(self.p_u(h_c), 2, -1)
        u_c = (mu_u_c
               + F.softplus(0.5 * logvar_u_c) * torch.randn_like(logvar_u_c))

        mu_u_t, logvar_u_t = torch.chunk(self.p_u(h_t), 2, -1)
        u_t = (mu_u_t
               + F.softplus(0.5 * logvar_u_t) * torch.randn_like(logvar_u_t))

        # Sample A
        _, bipartite = self.p_ga(u_c, u_t)
        bipartite = bipartite / (bipartite.sum(dim=-1, keepdim=True) + 1e-8)

        mu_qz_c, logvar_qz_c = torch.chunk(self.q_z(h_c), 2, -1)
        mu_qy_c, logvar_qy_c = torch.chunk(self.g_y(y_context), 2, -1)

        # Parameter of p(z): (b, num_t, z_dim)
        mu_pz_c = bipartite.matmul(mu_qy_c + mu_qz_c)
        logvar_pz_c = bipartite.matmul(logvar_qy_c + logvar_qz_c)

        # Sample z ~ p(z|par(R, y_R))
        z_c = (
            mu_pz_c
            + (F.softplus(0.5 * logvar_pz_c)) * torch.randn_like(logvar_pz_c))

        # Decode y: p(y|z)
        mu_py_t, logvar_py_t = torch.chunk(self.p_y(z_c), 2, dim=-1)
        var_py_t = F.softplus(logvar_py_t)

        return mu_py_t, var_py_t

    def loss_func(self, x_context: Tensor, y_context: Tensor, x_target: Tensor,
                  y_target: Tensor) -> Dict[str, Tensor]:
        """Loss function for ELBO.

        Notation comparison with original paper.

        * R: Context
        * M: Target
        * D = M + R: Dataset

        Args:
            x_context (torch.Tensor): x for context, size
                `(batch_size, num_context, x_dim)`.
            y_context (torch.Tensor): y for context, size
                `(batch_size, num_context, y_dim)`.
            x_target (torch.Tensor): x for target, size
                `(batch_size, num_target, x_dim)`.
            y_target (torch.Tensor): y for target, size
                `(batch_size, num_target, y_dim)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated loss.
        """

        # Representation
        h_c = self.f_x(x_context)
        h_t = self.f_x(x_target)

        # Sample u
        mu_u_c, logvar_u_c = torch.chunk(self.p_u(h_c), 2, -1)
        u_c = (mu_u_c
               + F.softplus(0.5 * logvar_u_c) * torch.randn_like(logvar_u_c))

        mu_u_t, logvar_u_t = torch.chunk(self.p_u(h_t), 2, -1)
        u_t = (mu_u_t
               + F.softplus(0.5 * logvar_u_t) * torch.randn_like(logvar_u_t))

        # Sample G, A
        graph, bipartite = self.p_ga(u_c, u_t)
        graph = graph / (graph.sum(dim=-1, keepdim=True) + 1e-8)
        bipartite = bipartite / (bipartite.sum(dim=-1, keepdim=True) + 1e-8)

        # Sample z: q(z|x)
        mu_qz_c, logvar_qz_c = torch.chunk(self.q_z(h_c), 2, -1)
        z_c = (mu_qz_c
               + F.softplus(0.5 * logvar_qz_c) * torch.randn_like(logvar_qz_c))

        mu_qz_t, logvar_qz_t = torch.chunk(self.q_z(h_t), 2, -1)
        z_t = (mu_qz_t
               + F.softplus(0.5 * logvar_qz_t) * torch.randn_like(logvar_qz_t))

        # Embed labels
        mu_qy_c, logvar_qy_c = torch.chunk(self.g_y(y_context), 2, -1)

        # Encode latents from graph: p(z|par(x, y))
        mu_pz_c = graph.matmul(mu_qy_c + mu_qz_c)
        logvar_pz_c = graph.matmul(logvar_qy_c + logvar_qz_c)

        mu_pz_t = bipartite.matmul(mu_qy_c + mu_qz_c)
        logvar_pz_t = bipartite.matmul(logvar_qy_c + logvar_qz_c)

        # Calculate KL loss: -E_{q(z|x)}[log p(z|x, y) - log q(z|x)]
        kl_pqz_c = (nll_normal(z_c, mu_pz_c, F.softplus(logvar_pz_c))
                    - nll_normal(z_c, mu_qz_c, F.softplus(logvar_qz_c)))
        kl_pqz_t = (nll_normal(z_t, mu_pz_t, F.softplus(logvar_pz_t))
                    - nll_normal(z_t, mu_qz_t, F.softplus(logvar_qz_t)))

        # Decode y
        mu_py_c, logvar_py_c = torch.chunk(self.p_y(z_c), 2, dim=-1)
        mu_py_t, logvar_py_t = torch.chunk(self.p_y(z_t), 2, dim=-1)

        # NLL loss: -E_{q(z|x)}[log p(y|z)]
        nll_py_c = nll_normal(y_context, mu_py_c, F.softplus(logvar_py_c))
        nll_py_t = nll_normal(y_target, mu_py_t, F.softplus(logvar_py_t))

        # Loss
        loss_c = (kl_pqz_c + nll_py_c).mean()
        loss_t = (kl_pqz_t + nll_py_t).mean()

        loss_dict = {
            "loss": loss_c + loss_t,
            "nll_loss_c": nll_py_c.mean(),
            "kl_loss_c": kl_pqz_c.mean(),
            "nll_loss_t": nll_py_t.mean(),
            "kl_loss_t": kl_pqz_t.mean(),
        }

        return loss_dict
