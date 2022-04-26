import torch
import torch.nn as nn

from collections import OrderedDict


class Model(nn.Module):
    """
    Model instantiation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def initialize(self, m):
        """
        Kaiming initialization for 1D convolutions.
        -- m: model to intantiate.
        """
        if isinstance(m, (nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")


class SampleCNN(Model):
    """
    Model used in CLMR for Self-Supervised learning.
    """

    def __init__(self, strides, supervised, out_dim):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.supervised = supervised
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"

        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

        if self.supervised:
            self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        """
        Pass waveform through model.
        -- x: waveform
        """

        out = self.sequential(x)

        if self.supervised:
            out = self.dropout(out)

        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        return logit


def load_encoder_checkpoint(checkpoint_path, output_dim=50):
    """
    Load state dictionary from checkpoint file.
    """
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if "encoder." in k:
            new_state_dict[k.replace("encoder.", "")] = v

    new_state_dict["fc.weight"] = torch.zeros(output_dim, 512)
    new_state_dict["fc.bias"] = torch.zeros(output_dim)

    return new_state_dict


class CLMR(nn.Module):
    """
    Use of CLMR model for beat tracking.
    https://github.com/Spijkervet/CLMR
    """

    def __init__(self, pretrained=False, PATH="models/saved/clmr_epoch=10000.ckpt"):
        super(CLMR, self).__init__()

        encoder = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=True,
            out_dim=50,
        )

        if pretrained:
            state_dict = load_encoder_checkpoint(PATH, 50)
            encoder.load_state_dict(state_dict)

        encoder = encoder.sequential
        self.encoder = nn.Sequential(*list(encoder.children())[:-6])

        # Output
        self.fc1 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Pass waveform through model.
        """
        x = self.encoder(x)

        # Flatten layer
        x = self.fc1(x)
        x = self.sig(x)
        x = x.reshape(x.size(0), x.size(1) * x.size(2))

        return x
