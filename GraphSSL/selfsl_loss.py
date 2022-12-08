import torch

### Self Supervised Loss ###

def infonce(readout_anchor, readout_positive, tau=0.5, norm=True):
    """
    The InfoNCE (NT-XENT) loss in contrastive learning. The implementation
    follows the paper `A Simple Framework for Contrastive Learning of 
    Visual Representations <https://arxiv.org/abs/2002.05709>`.
    Args:
        readout_anchor, readout_positive: Tensor of shape [batch_size, feat_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    """
    #print(readout_anchor.shape, readout_positive.shape)

    batch_size = readout_anchor.shape[0]
    sim_matrix = torch.einsum("ik,jk->ij", readout_anchor, readout_positive)

    if norm:
        readout_anchor_abs = readout_anchor.norm(dim=1)
        readout_positive_abs = readout_positive.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum("i,j->ij", readout_anchor_abs, readout_positive_abs)

    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

def get_expectation(masked_d_prime, positive=True):
    """
    Args:
        masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global,
                        tensor of shape [n_nodes, n_graphs] for local_global.
        positive (bool): Set True if the d_prime is masked for positive pairs,
                        set False for negative pairs.
    """

    log_2 = np.log(2.)
    if positive:
        score = log_2 - F.softplus(-masked_d_prime)
    else:
        score = F.softplus(-masked_d_prime) + masked_d_prime - log_2
    return score

def jensen_shannon(readout_anchor, readout_positive):
    """
    The Jensen-Shannon Estimator of Mutual Information used in contrastive learning. The
    implementation follows the paper `Learning deep representations by mutual information 
    estimation and maximization <https://arxiv.org/abs/1808.06670>`.
    Note: The JSE loss implementation can produce negative values because a :obj:`-2log2` shift is 
        added to the computation of JSE, for the sake of consistency with other f-convergence losses.
    Args:
        readout_anchor, readout_positive: Tensor of shape [batch_size, feat_dim].
    """

    batch_size = readout_anchor.shape[0]

    pos_mask = torch.zeros((batch_size, batch_size))
    neg_mask = torch.ones((batch_size, batch_size))
    for graphidx in range(batch_size):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    d_prime = torch.matmul(readout_anchor, readout_positive.t())

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / batch_size
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (batch_size * (batch_size - 1))
    return E_neg - E_pos