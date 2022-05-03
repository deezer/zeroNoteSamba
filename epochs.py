import torch

from processing.evaluate import beat_tracking as eval


def train_epoch(
    model, criterion, optimizer, _status, indices, real_times, inputs, masks, threshold
):
    """
    Training epoch.
    -- model : model to train
    -- criterion : loss function
    -- optimizer : self-explanatory
    -- _status : pretrained model?
    -- indices : set indices
    -- real_times : real beat times
    -- inputs : spectrograms of audio to feed to NN
    -- masks : beat activation functions
    -- threshold : threshold value for evaluation
    """
    full_loss, f_measure, cmlc, cmlt, amlc, amlt, info_gain = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    model.train()

    for batch_idx, wav in enumerate(indices):
        times = real_times[wav]

        if _status == "pretrained":
            vqt = inputs[wav]
            vqt1 = torch.reshape(
                vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])
            ).cuda()
            vqt2 = torch.reshape(
                vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])
            ).cuda()

            msk = masks[wav]
            msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

            optimizer.zero_grad()

            # Apply model to full batch
            output = model(vqt1, vqt2)

            loss = criterion(output, msk)
            loss.backward()
            optimizer.step()

        else:
            vqt = inputs[wav]
            vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

            msk = masks[wav]
            msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

            optimizer.zero_grad()

            # Apply model to full batch
            output = model(vqt)

            loss = criterion(output, msk)
            loss.backward()
            optimizer.step()

        full_loss += loss.item()

        cpu_output = output.squeeze(0).cpu().detach().numpy()

        res = eval(cpu_output, times, threshold=threshold)
        f_measure += res[0]
        cmlc += res[1]
        cmlt += res[2]
        amlc += res[3]
        amlt += res[4]
        info_gain += res[5]

    full_loss = full_loss / (batch_idx + 1)
    f_measure = f_measure / (batch_idx + 1)
    cmlc = cmlc / (batch_idx + 1)
    cmlt = cmlt / (batch_idx + 1)
    amlc = amlc / (batch_idx + 1)
    amlt = amlt / (batch_idx + 1)
    info_gain = info_gain / (batch_idx + 1)

    return model, optimizer, full_loss, f_measure, cmlc, cmlt, amlc, amlt, info_gain


def val_epoch(model, criterion, _status, indices, real_times, inputs, masks, threshold):
    """
    Validation epoch.
    -- model : model to train
    -- criterion : loss function
    -- _status : pretrained model?
    -- indices : set indices
    -- real_times : real beat times
    -- inputs : signals or spectrograms of audio to feed to NN
    -- masks : beat activation functions
    -- threshold : threshold value for evaluation
    """
    full_loss, f_measure, cmlc, cmlt, amlc, amlt, info_gain = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    model.eval()

    for batch_idx, wav in enumerate(indices):
        with torch.no_grad():
            times = real_times[wav]

            if _status == "pretrained":
                vqt = inputs[wav]
                vqt1 = torch.reshape(
                    vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])
                ).cuda()
                vqt2 = torch.reshape(
                    vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])
                ).cuda()

                msk = masks[wav]
                msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                output = model(vqt1, vqt2)

                loss = criterion(output, msk)

            else:
                vqt = inputs[wav]
                vqt = torch.reshape(
                    vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])
                ).cuda()

                msk = masks[wav]
                msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                # Apply model to full batch
                output = model(vqt)

                loss = criterion(output, msk)

            full_loss += loss.item()

            cpu_output = output.squeeze(0).cpu().detach().numpy()

            res = eval(cpu_output, times, threshold=threshold)
            f_measure += res[0]
            cmlc += res[1]
            cmlt += res[2]
            amlc += res[3]
            amlt += res[4]
            info_gain += res[5]

    full_loss = full_loss / (batch_idx + 1)
    f_measure = f_measure / (batch_idx + 1)
    cmlc = cmlc / (batch_idx + 1)
    cmlt = cmlt / (batch_idx + 1)
    amlc = amlc / (batch_idx + 1)
    amlt = amlt / (batch_idx + 1)
    info_gain = info_gain / (batch_idx + 1)

    return full_loss, f_measure, cmlc, cmlt, amlc, amlt, info_gain
