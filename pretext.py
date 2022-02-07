import os
import yaml
import random
import torch
import random
import shutil
import pickle

import numpy as np
import librosa as audio_lib
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from random import randint
from torch.utils.data import TensorDataset, DataLoader

# File imports
import processing.stem_check as stem_check
import processing.input_rep as input_rep

from models.models import Pretext_CNN
from models.loss_functions import NTXent

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

plt.rcParams["figure.figsize"] = (15, 5)


def drum_anchor_positive(stems, ymldict):
    """
    Function for generating anchor and positive samples.
    -- stems       : dictionary of stem names and signals
    -- ymldict     : YAML dictionary
    """
    rms_bool = False

    # Load YAML variables
    length = ymldict.get("clip_len")
    mode = ymldict.get("input_mode")
    lower_p = ymldict.get("lower_p")
    upper_p = ymldict.get("upper_p")

    idx = 0

    anchor = None

    for name, sig in stems.items():
        if name == "drums":
            possignal = np.zeros(len(sig))
            possignal[:] = sig[:]

        else:
            if anchor is None:
                anchor = np.zeros(len(sig))
                anchor[:] = sig[:]

            else:
                anchor[:] += sig[:]

    while rms_bool == False:
        # Largest possible stop sample index
        stop = len(anchor) - length * 16000 - 1

        # Generate start indices
        ran = randint(0, stop)

        temp_anchor = anchor[ran : ran + length * 16000]
        temp_possignal = possignal[ran : ran + length * 16000]

        rms_bool = stem_check.check_CL_clips(
            temp_anchor, temp_possignal, lower_p, upper_p
        )

        idx += 1

        if idx > 9:
            lower_p = lower_p / 2

    anchor_cqt = input_rep.generate_XQT(temp_anchor, 16000, mode)
    possignal_cqt = input_rep.generate_XQT(temp_possignal, 16000, mode)

    return anchor, possignal, anchor_cqt, possignal_cqt


def create_memory_bank(number_of_samples, ymldict, fps, pkl_fp):
    """
    Function for creationg a memory bank of anchors and their positives.
    -- number_of_samples : length of memory bank
    -- ymldict           : YAML dictionary
    -- fps               : file paths list
    -- pkl_fp            : pkl file name
    """
    # File paths
    random.shuffle(fps)

    new_fps = []
    pr_fps = []

    all_stems = {}

    print("Loading files into dictionary...")

    idx = 0
    for fp in tqdm(fps):
        temp = {}
        temp["bass"], _ = audio_lib.load("new_data/" + fp + "/bass.wav", sr=None)
        temp["drums"], _ = audio_lib.load("new_data/" + fp + "/drums.wav", sr=None)
        temp["other"], _ = audio_lib.load("new_data/" + fp + "/other.wav", sr=None)
        temp["vocals"], _ = audio_lib.load("new_data/" + fp + "/vocals.wav", sr=None)

        if len(temp["vocals"]) < 16000 * 10:
            print("File path {} is problematic.".format(fp))
            shutil.rmtree("new_data/" + fp)
            pr_fps.append(fp)
            continue

        all_stems[fp] = temp

        idx += 1
        new_fps.append(fp)

        if idx == number_of_samples:
            print("Done.")
            break

    # Initialize for size of bank
    fp = new_fps.pop()
    fps.remove(fp)

    temp_stems = all_stems[fp.strip()]

    _, _, anchor_cqt, possignal_cqt = drum_anchor_positive(temp_stems, ymldict)

    bank = np.zeros((number_of_samples, 2, anchor_cqt.shape[0], anchor_cqt.shape[1]))
    bank[0, 0, :, :] = anchor_cqt[:, :]
    bank[0, 1, :, :] = possignal_cqt[:, :]

    x = 1

    print("Creating anchors and positives...")

    # Fill up bank
    for fp in tqdm(new_fps):
        fps.remove(fp)

        # Run Spleeter
        temp_stems = all_stems[fp.strip()]

        _, _, anchor_cqt, possignal_cqt = drum_anchor_positive(temp_stems, ymldict)

        bank[x, 0, :, :] = anchor_cqt[:, :]
        bank[x, 1, :, :] = possignal_cqt[:, :]

        x += 1

    for fp in pr_fps:
        fps.remove(fp)

    with open(pkl_fp, "wb") as handle:
        pickle.dump(bank, handle, pickle.HIGHEST_PROTOCOL)
        print("-- Saved file {}. --".format(pkl_fp))

    return bank, all_stems, fps


def train_model(ymldict, saved=True):
    """
    Function for training a model.
    Steps include batch creation and calls to training epoch.
    -- ymldict   : YAML parameters
    -- saved     : whether pkl files have been saved
    """
    # Load YAML parameters
    val_len = ymldict.get("val_len")
    train_pkl = ymldict.get("train_pkl")
    batch_len = ymldict.get("batch_size")
    epochs = ymldict.get("num_epochs")
    lr = ymldict.get("lr")
    tmp = ymldict.get("temp")

    fps = os.listdir("new_data/")
    random.shuffle(fps)

    # Model, optimizer, criterion...
    criterion = NTXent(batch_len=batch_len, temperature=tmp).to(device1)

    model = Pretext_CNN(pretext=True)
    model.anchor.to(device0)
    model.postve.to(device1)
    model_name = "shift_pret_cnn_{}.pth".format(batch_len)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.75, verbose=False
    )

    train_losses = []
    train_an_pos = []
    train_an_neg = []
    val_losses = []
    val_an_pos = []
    val_an_neg = []

    best_val_loss = np.inf

    # Train model
    for epoch in range(epochs):

        print("\n--- Epoch {} ---".format(epoch))

        # Create data loader
        if epoch == 0:
            print("Creating memory bank...")

            if saved == False:
                print("Saving pkl files!")

                _, _, fps = create_memory_bank(
                    val_len, ymldict, fps, "data/Validation/val_bank.pkl"
                )

                for xx in trange(10):
                    _, _, fps = create_memory_bank(
                        train_pkl,
                        ymldict,
                        fps,
                        "data/Train/train_bank_{}.pkl".format(xx),
                    )

                print("Number of files remaining is {}.".format(len(fps)))
                quit()

            else:
                print("Loading pkl files...")

                train_bank = np.zeros((28800, 2, 96, 626), dtype=np.float32)
                val_bank = np.zeros((6400, 2, 96, 626), dtype=np.float32)

                for xx in trange(10):
                    with open(
                        "data/Train/train_bank_{}.pkl".format(xx), "rb"
                    ) as handle:
                        train_bank[xx * 2880 : xx * 2880 + 2880, :, :, :] = pickle.load(
                            handle
                        )[:, :, :, :]

                with open("data/Validation/val_bank.pkl", "rb") as handle:
                    val_bank[:, :, :, :] = pickle.load(handle)[:, :, :, :]

        np.random.shuffle(train_bank)

        if epoch == 0:
            new_val_bank = np.zeros((6400 * batch_len, 2, 96, 313), dtype=np.float32)

            print("Creating new validation shifts...")
            for xx in trange(6400):
                randomlist = random.sample(range(0, 313), batch_len)

                for ii, start_idx in enumerate(randomlist):
                    new_val_bank[xx * batch_len + ii, :, :, :] = val_bank[
                        xx, :, :, start_idx : start_idx + 313
                    ]

        full_train_loss = 0.0
        full_train_anpos = 0.0
        full_train_anneg = 0.0
        full_val_loss = 0.0
        full_val_anpos = 0.0
        full_val_anneg = 0.0

        for jj in range(20):
            new_train_bank = np.zeros((1440 * batch_len, 2, 96, 313), dtype=np.float32)

            print("\n{} : Creating new training shifts...".format(jj))
            for xx in trange(1440):
                randomlist = random.sample(range(0, 313), batch_len)

                for ii, start_idx in enumerate(randomlist):
                    new_train_bank[xx * batch_len + ii, :, :, :] = train_bank[
                        jj * 1440 + xx, :, :, start_idx : start_idx + 313
                    ]

            train_ds = TensorDataset(torch.tensor(new_train_bank).float())
            train_loader = DataLoader(train_ds, batch_size=batch_len, shuffle=False)

            # Train epoch
            print("{} : Training...".format(jj))
            model, temp_train_loss, temp_train_anpos, temp_train_anneg = train_epoch(
                model, train_loader, criterion, optimizer
            )

            full_train_loss += temp_train_loss
            full_train_anpos += temp_train_anpos
            full_train_anneg += temp_train_anneg

        full_train_loss /= 20
        full_train_anpos /= 20
        full_train_anneg /= 20

        train_losses.append(full_train_loss)
        train_an_pos.append(full_train_anpos)
        train_an_neg.append(full_train_anneg)

        print("\n!!! Mean training batch loss is {:.3f}.".format(full_train_loss))
        print(
            "!!! Mean training anchor / positive similiarity is {:.3f}.".format(
                full_train_anpos
            )
        )
        print(
            "!!! Mean training anchor / negative similiarity is {:.3f}.".format(
                full_train_anneg
            )
        )

        print("\n{} : Validating...".format(epoch))

        for zz in trange(10):
            val_ds = TensorDataset(
                torch.tensor(
                    new_val_bank[
                        640 * batch_len * zz : 640 * batch_len * (zz + 1), :, :, :
                    ]
                ).float()
            )
            val_loader = DataLoader(val_ds, batch_size=batch_len, shuffle=False)

            temp_val_loss, temp_val_anpos, temp_val_anneg = val_epoch(
                model, val_loader, criterion, optimizer
            )

            full_val_loss += temp_val_loss
            full_val_anpos += temp_val_anpos
            full_val_anneg += temp_val_anneg

        full_val_loss /= 10
        full_val_anpos /= 10
        full_val_anneg /= 10

        print("\n!!! Mean validation batch loss is {:.3f}.".format(full_val_loss))
        print(
            "!!! Mean validation anchor / positive similiarity is {:.3f}.".format(
                full_val_anpos
            )
        )
        print(
            "!!! Mean validation anchor / negative similiarity is {:.3f}.".format(
                full_val_anneg
            )
        )

        # Save model
        if full_val_loss < best_val_loss:
            torch.save(model.state_dict(), "models/" + model_name)
            print("...Saved model to " + "models/" + model_name)
            best_val_loss = full_val_loss

        val_losses.append(full_val_loss)
        val_an_pos.append(full_val_anpos)
        val_an_neg.append(full_val_anneg)

        scheduler.step()

        if int(epoch + 1) % 5 == 0:
            os.makedirs("figures", exist_ok=True)

            plt.figure()
            plt.plot(train_losses)
            plt.plot(val_losses)
            plt.legend(["Train", "Validation"])
            plt.title("Mean Batch Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig("figures/shift_loss_16.pdf", dpi=300, format="pdf")

            plt.figure()
            plt.plot(train_an_pos)
            plt.plot(train_an_neg)
            plt.legend(["Anchors & Positives", "Anchors & Negatives"])
            plt.title("Train Set Mean Batch Similarity")
            plt.ylim([0, 1])
            plt.xlabel("Epochs")
            plt.ylabel("Cosine Similarity")
            plt.savefig("figures/shift_train_similarity_16.pdf", dpi=300, format="pdf")

            plt.figure()
            plt.plot(val_an_pos)
            plt.plot(val_an_neg)
            plt.legend(["Anchors & Positives", "Anchors & Negatives"])
            plt.title("Validation Set Mean Batch Similarity")
            plt.ylim([0, 1])
            plt.xlabel("Epochs")
            plt.ylabel("Cosine Similarity")
            plt.savefig("figures/shift_val_similarity_16.pdf", dpi=300, format="pdf")

    return model


def train_epoch(model, train_loader, criterion, optimizer):
    """
    Function for CL model training.
    -- model        : model to train
    -- train_loader : loader with batches that contain 1 anchor, 1 positive, and negatives
    -- criterion    : loss function
    -- optimizer    : optimizer defined
    """
    full_train_loss = 0.0
    full_train_anpos = 0.0
    full_train_anneg = 0.0

    model.train()

    for batch_idx, [batch] in enumerate(tqdm(train_loader)):
        anchors = batch[:, 0:1, :, :].to(device0)
        postves = batch[:, 1:2, :, :].to(device1)

        optimizer.zero_grad()

        # Apply model to full batch
        anc_emb, pos_emb = model(anchors, postves)

        anc_emb = anc_emb.to(device1)

        loss, sim_an_pos, sim_an_neg = criterion(anc_emb, pos_emb)
        loss.backward()
        optimizer.step()

        full_train_loss += loss.item()
        full_train_anpos += sim_an_pos
        full_train_anneg += sim_an_neg

    full_train_loss = full_train_loss / (batch_idx + 1)
    full_train_anpos = full_train_anpos / (batch_idx + 1)
    full_train_anneg = full_train_anneg / (batch_idx + 1)

    print("*** Mean training batch loss is {:.3f}.".format(full_train_loss))
    print(
        "*** Mean training anchor / positive similiarity is {:.3f}.".format(
            full_train_anpos
        )
    )
    print(
        "*** Mean training anchor / negative similiarity is {:.3f}.".format(
            full_train_anneg
        )
    )

    return model, full_train_loss, full_train_anpos, full_train_anneg


def val_epoch(model, val_loader, criterion, optimizer):
    """
    Function for CL model training.
    -- model        : model to train
    -- val_loader   : loader with batches that contain 1 anchor, 1 positive, and negatives
    -- criterion    : loss function
    -- optimizer    : optimizer defined
    """
    full_val_loss = 0.0
    full_val_anpos = 0.0
    full_val_anneg = 0.0

    model.eval()

    for batch_idx, [batch] in enumerate(val_loader):
        with torch.no_grad():
            anchors = batch[:, 0:1, :, :].to(device0)
            postves = batch[:, 1:2, :, :].to(device1)

            optimizer.zero_grad()

            # Apply model to full batch
            anc_emb, pos_emb = model(anchors, postves)

            anc_emb = anc_emb.to(device1)

            loss, sim_an_pos, sim_an_neg = criterion(anc_emb, pos_emb)

            full_val_loss += loss.item()
            full_val_anpos += sim_an_pos
            full_val_anneg += sim_an_neg

    full_val_loss = full_val_loss / (batch_idx + 1)
    full_val_anpos = full_val_anpos / (batch_idx + 1)
    full_val_anneg = full_val_anneg / (batch_idx + 1)

    return full_val_loss, full_val_anpos, full_val_anneg


if __name__ == "__main__":
    # Load YAML file configurations
    stream = open("configuration/config.yaml", "r")
    ymldict = yaml.safe_load(stream)

    _ = train_model(ymldict, saved=True)
