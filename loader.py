import torch

from models.models import DS_CNN, Down_CNN


def load_models(_status, _pre, _lr):
    """
    Function for loading loss, optimizer, and model.
    -- _status : pretrained or vanilla?
    -- _pre    : frozen weights
    -- _lr     : learning rate
    """
    # Set loss function
    criterion = torch.nn.BCELoss().cuda()

    # Set model and pre-trained layers if need be; optimizer set accordingly
    if _status == "pretrained":
        print("\nPretrained learning mode...")

        model = Down_CNN().cuda()
        state_dict = torch.load(
            "models/saved/shift_pret_cnn_16.pth", map_location=torch.device("cuda")
        )
        model.pretext.load_state_dict(state_dict)

        if _pre == "frozen":
            for param in model.pretext.anchor.pretrained.parameters():
                param.requires_grad = False

            for param in model.pretext.postve.pretrained.parameters():
                param.requires_grad = False

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=_lr,
                betas=(0.9, 0.999),
            )

        elif _pre == "finetune":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.5 * _lr * 10e-2, betas=(0.9, 0.999)
            )

    else:
        print("\nVanilla learning mode...")
        model = DS_CNN().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=_lr, betas=(0.9, 0.999))

    return criterion, optimizer, model