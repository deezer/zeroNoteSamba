import torch
import torch.nn as nn


class _CNN(nn.Module):
    """
    Convolutional and recurrent layers.
    """
    def __init__(self):
        super(_CNN, self).__init__()

        # Inputs are size 8 * 9000
        self.cv1 = nn.Conv2d(in_channels=1  , out_channels=64 , kernel_size=(3, 11), padding=(1, 5 ))
        self.cv2 = nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=(7, 13), padding=(3, 6 ))
        self.cv3 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=(5, 15), padding=(2, 7 ))
        self.cv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(9, 17), padding=(4, 8 ))
        self.cv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 19), padding=(1, 9 ))
        self.cv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 21), padding=(2, 10))
        self.cv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 23), padding=(0, 11))
        self.cv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 25), padding=(0, 12))

        self.relu   = nn.ReLU(inplace=True)
        self.maxpl1 = nn.MaxPool2d((3, 1), padding=(0, 0))
        self.maxpl2 = nn.MaxPool2d((4, 1), padding=(0, 0))
        self.maxpl3 = nn.MaxPool2d((8, 1), padding=(0, 0))

        self.dp = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        Pass input through convolutional layers.
        -- x : input (vqt)
        """
        out = self.cv1(x)
        out = self.relu(out)
        out = self.dp(out)

        out = self.cv2(out)
        out = self.maxpl1(out)
        out = self.relu(out)
        out = self.dp(out)

        out = self.cv3(out)
        out = self.relu(out)
        out = self.dp(out)

        out = self.cv4(out)
        out = self.maxpl2(out)
        out = self.relu(out)
        out = self.dp(out)

        out = self.cv5(out)
        out = self.relu(out)
        out = self.dp(out)

        out = self.cv6(out)
        out = self.maxpl3(out)
        out = self.relu(out)
        out = self.dp(out)

        out = self.cv7(out)
        out = self.relu(out)
        out = self.dp(out)

        out = self.cv8(out)
        out = self.relu(out)
        out = self.dp(out)
        
        out = torch.squeeze(out, dim=2)
        
        return out


class DS_CNN(nn.Module):
    """
    Fully-convolutional architecture for beat tracking.
    """
    def __init__(self):
        super(DS_CNN, self).__init__()

        self.pretrained = _CNN()

        # Output
        self.fc1 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Pass input through convolutional and recurrent layers.
        -- x : input (vqt)
        """
        x = self.pretrained(x)

        # Flatten layer
        x = self.fc1(x)
        x = self.sig(x)
        x = x.reshape(x.size(0), x.size(1) * x.size(2))

        return x


class Pretext_CNN(nn.Module):
    """
    DS_CNN tailored for percussive and non-percussive stems.
    """
    def __init__(self):
        super(Pretext_CNN, self).__init__()

        self.anchor = DS_CNN()
        self.postve = DS_CNN()
        
    def forward(self, anc, pos):
        """
        Pass vqts through each model.
        """
        anc_emb = self.anchor(anc)
        pos_emb = self.postve(pos)
        
        return anc_emb, pos_emb


class Down_CNN(nn.Module):
    """
    Use of Pretext_CNN for downstream tasks.
    """
    def __init__(self, reduction="max"):
        super(Down_CNN, self).__init__()

        self.pretext   = Pretext_CNN()
        self.reduction = reduction

    def forward(self, anc, pos):
        """
        Pass each input through each model. Add and output.
        """
        anc_emb, pos_emb = self.pretext(anc, pos)

        if (self.reduction == "mean"):
            emb = torch.div(anc_emb + pos_emb, 2)

        else:
            emb = torch.maximum(anc_emb, pos_emb)

        return emb