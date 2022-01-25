import torch
import torch.nn as nn


class NTXent(nn.Module):
    """
    Compute NT-Xent loss for contrastive learning.
    """
    def __init__(self, batch_len, temperature=.25, in_features=626, mode='CS'):
        """
        Arguments:
        -- batch_len   : batch size
        -- temperature : parameter
        -- in_features : input vector dimensions
        -- mode        : cosine or bilinear similarity
        """
        super(NTXent, self).__init__()

        self.batch_len   = batch_len
        self.mode        = mode
        self.temperature = temperature

        if (mode == 'CS'):
            self.CS = nn.CosineSimilarity(dim=1, eps=1e-08)

        else:
            self.CS = nn.Bilinear(in_features, in_features, 1, bias=False)


    def forward(self, anchors, poss):
        """
        Arguments:
        -- anchors : tensor of shape (batch_len, embedding_size)
        -- poss    : tensor of shape (batch_len, embedding_size)
        """
        full_loss  = torch.zeros(self.batch_len)

        cos_an_pos = 0.
        cos_an_neg = 0.

        for xx in range(anchors.shape[0]):
            anchor = anchors[xx:xx+1, :]
            pos    = poss[xx:xx+1, :]

            sim_an_pos = self.CS(anchor, pos)
            cos_an_pos += sim_an_pos.item()
            sim_an_pos = torch.div(sim_an_pos, self.temperature)
            sim_num    = sim_an_pos.exp()

            sim_an_png = self.CS(anchor, poss)
            cos_an_neg += (sim_an_png.sum().item() - sim_an_png[xx].item()) / (self.batch_len - 1)
            sim_an_png = torch.div(sim_an_png, self.temperature)
            sim_den    = sim_an_png.exp().sum()

            full_loss[xx] = -torch.log(torch.div(sim_num, sim_den))

        return full_loss.mean() , cos_an_pos / self.batch_len, cos_an_neg / self.batch_len