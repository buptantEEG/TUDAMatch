import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from abc import abstractmethod
import math
import sys
sys.path.append("..")
from .attention import Seq_Transformer

class Discriminator_AR(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, configs):
        """Init discriminator."""
        self.input_dim = configs.out_channels
        super(Discriminator_AR, self).__init__()

        self.AR_disc = nn.GRU(input_size=self.input_dim, hidden_size=configs.disc_AR_hid,num_layers = configs.disc_n_layers,bidirectional=configs.disc_AR_bid, batch_first=True)
        self.DC = nn.Linear(configs.disc_AR_hid+configs.disc_AR_hid*configs.disc_AR_bid, 1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, self.input_dim )
        encoder_outputs, (encoder_hidden) = self.AR_disc(input)
        features = encoder_outputs[:, -1, :]
        domain_output = self.DC(features)
        return domain_output
    def get_parameters(self):
        parameter_list = [{"params":self.AR_disc.parameters(), "lr_mult":0.01, 'decay_mult':1}, {"params":self.DC.parameters(), "lr_mult":0.01, 'decay_mult':1},]
        return parameter_list

class Discriminator_ATT(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, configs):
        """Init discriminator."""
        self.patch_size =  configs.patch_size
        self.hid_dim = configs.att_hid_dim
        self.depth= configs.depth
        self.heads = configs.heads
        self.mlp_dim = configs.mlp_dim
        super(Discriminator_ATT, self).__init__()
        self.transformer= Seq_Transformer(patch_size=self.patch_size, dim=configs.att_hid_dim, depth=self.depth, heads= self.heads , mlp_dim=self.mlp_dim)
        self.DC = nn.Linear(configs.att_hid_dim, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input):
        # print(f"disc input:{input.shape}")
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        # print(input.size)
        input = input.view(input.size(0),-1, self.patch_size )
        # input = torch.rand_like(input)
        features = self.transformer(input)
        domain_output = self.DC(features)
        domain_output = self.sigmoid(domain_output)
        return domain_output

        
class Discriminator(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.feat_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 1),
            # nn.Linear(configs.feat_dim, 1),
            nn.Sigmoid()
            # nn.LogSoftmax(dim=1)
        )
    def forward(self, input):
        """Forward the discriminator."""
        # print(f"disc input:{input.shape}")
        out = self.layer(input)
        # print(out)
        return out