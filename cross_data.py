import yaml
import torch
import random
import pickle
import numpy as np

from models.models       import DS_CNN, Down_CNN
from processing.evaluate import beat_tracking as eval


def train_model(train_wavs, train_vqts, train_masks, train_real_times, 
                test_wavs , test_vqts , test_masks , test_real_times , 
                ymldict):
    """
    Function for training model on Hainsworth data set.
    -- train_wavs       : list of training wav files (dictionary keys)
    -- train_vqts       : training spectrograms of audio
    -- train_masks      : training beat tracking pulse vectors
    -- train_real_times : training list with real beat tracking times in seconds
    -- test_wavs        : list of test wav files (dictionary keys)
    -- test_vqts        : test spectrograms of audio
    -- test_masks       : test tracking pulse vectors
    -- test_real_times  : test list with real beat tracking times in seconds
    -- ymldict          : YAML parameters
    """
    # Load the Hainsworth stuff:
    _status    = ymldict.get("cross_status")
    _pre       = ymldict.get("cross_pre")
    _train_set = ymldict.get("cross_train_set")

    random.shuffle(train_wavs)

    _lr   = ymldict.get("cross_lr")
    _eval = ymldict.get("cross_eval")

    threshold = False

    if (_eval == 'threshold'):
        threshold = True

    cv_len = len(train_wavs) / 8

    split1  = train_wavs[0:round(cv_len)]
    split2  = train_wavs[round(cv_len):round(cv_len * 2)]
    split3  = train_wavs[round(cv_len * 2):round(cv_len * 3)]
    split4  = train_wavs[round(cv_len * 3):round(cv_len * 4)]
    split5  = train_wavs[round(cv_len * 4):round(cv_len * 5)]
    split6  = train_wavs[round(cv_len * 5):round(cv_len * 6)]
    split7  = train_wavs[round(cv_len * 6):round(cv_len * 7)]
    split8  = train_wavs[round(cv_len * 7):]

    splits  = [split1, split2, split3, split4, split5, split6, split7, split8]

    f1   = []
    cmlc = []
    cmlt = []
    amlc = []
    amlt = []
    ig   = []

    for jj in range(8):

        # Set loss function
        criterion = torch.nn.BCELoss().cuda()

        # Set model and pre-trained layers if need be; optimizer set accordingly
        if (_status == "pretrained"):
            print("Pretrained learning mode...")

            model      = Down_CNN().cuda()
            state_dict = torch.load("models/saved/shift_pret_cnn_16.pth", map_location=torch.device('cuda'))
            model.pretext.load_state_dict(state_dict)

            if (_pre == "frozen"):
                for param in model.pretext.anchor.pretrained.parameters():
                    param.requires_grad = False

                for param in model.pretext.postve.pretrained.parameters():
                    param.requires_grad = False

                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                            lr=_lr, betas=(0.9, 0.999))

            elif (_pre == "finetune"):
                optimizer = torch.optim.Adam(model.parameters(), lr=0.5 * _lr * 10e-2, betas=(0.9, 0.999))

        else: 
            print("Vanilla learning mode...")
            model     = DS_CNN(pretext=True).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=_lr, betas=(0.9, 0.999))
    
        val_counter = 0

        train_loss = []
        val_loss   = []
        train_f1   = []
        val_f1     = []
        train_indices = []

        for ii in range(8):
            if (ii == jj):
                continue
            else:
                train_indices = train_indices + splits[ii]

        val_indices  = splits[jj]

        random.shuffle(train_indices)

        best_f1   = 0.

        # Train model
        for epoch in range(500):
            print("\n-- Epoch {} --".format(epoch))

            full_train_loss = 0.
            full_val_loss   = 0.
            train_f_measure, train_cmlc, train_cmlt, train_amlc, train_amlt, train_info_gain = 0., 0., 0., 0., 0., 0.
            val_f_measure  , val_cmlc  , val_cmlt  , val_amlc  , val_amlt  , val_info_gain   = 0., 0., 0., 0., 0., 0.

            model.train()
            
            for batch_idx, wav in enumerate(train_indices):
                times = train_real_times[wav]

                if (_status == 'pretrained'):
                    vqt  = train_vqts[wav]
                    vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                    vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()

                    msk = train_masks[wav]
                    msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                    optimizer.zero_grad()
                    
                    # Apply model to full batch
                    output = model(vqt1, vqt2)
                    
                    loss = criterion(output, msk)
                    loss.backward()
                    optimizer.step()

                else:
                    vqt = train_vqts[wav]
                    vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                    msk = train_masks[wav]
                    msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                    optimizer.zero_grad()

                    # Apply model to full batch
                    output = model(vqt)
                    
                    loss = criterion(output, msk)
                    loss.backward()
                    optimizer.step()
                
                full_train_loss += loss.item()

                cpu_output = output.squeeze(0).cpu().detach().numpy()

                res = eval(cpu_output, times, threshold=threshold)
                train_f_measure += res[0]
                train_cmlc      += res[1]
                train_cmlt      += res[2]
                train_amlc      += res[3]
                train_amlt      += res[4]
                train_info_gain += res[5]
            
            full_train_loss = full_train_loss / (batch_idx + 1)
            train_f_measure = train_f_measure / (batch_idx + 1)
            train_cmlc      = train_cmlc      / (batch_idx + 1)
            train_cmlt      = train_cmlt      / (batch_idx + 1)
            train_amlc      = train_amlc      / (batch_idx + 1)
            train_amlt      = train_amlt      / (batch_idx + 1)
            train_info_gain = train_info_gain / (batch_idx + 1)

            print('\nMean training loss is {:.3f}.'.format(full_train_loss))
            print('Mean F1-score is {:.3f}.'.format(train_f_measure))

            model.eval()

            for batch_idx, wav in enumerate(val_indices):
                with torch.no_grad():
                    times = train_real_times[wav]

                    if (_status == 'pretrained'):
                        vqt  = train_vqts[wav]
                        vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                        vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()     

                        msk = train_masks[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        output = model(vqt1, vqt2)
                        
                        loss = criterion(output, msk)

                    else:
                        vqt = train_vqts[wav]
                        vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                        msk = train_masks[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        # Apply model to full batch
                        output = model(vqt)
                        
                        loss = criterion(output, msk)
                    
                    full_val_loss += loss.item()

                    cpu_output = output.squeeze(0).cpu().detach().numpy()

                    res = eval(cpu_output, times, threshold=threshold)
                    val_f_measure += res[0]
                    val_cmlc      += res[1]
                    val_cmlt      += res[2]
                    val_amlc      += res[3]
                    val_amlt      += res[4]
                    val_info_gain += res[5]

            full_val_loss = full_val_loss / (batch_idx + 1)
            val_f_measure = val_f_measure / (batch_idx + 1)
            val_cmlc      = val_cmlc      / (batch_idx + 1)
            val_cmlt      = val_cmlt      / (batch_idx + 1)
            val_amlc      = val_amlc      / (batch_idx + 1)
            val_amlt      = val_amlt      / (batch_idx + 1)
            val_info_gain = val_info_gain / (batch_idx + 1)

            print('\nMean validation loss     is {:.3f}.'.format(full_val_loss))
            print('Mean validation F1-score is {:.3f}.'.format(val_f_measure))

            if (val_f_measure > best_f1):
                mod_fp = "models/saved/cross_{}_{}.pth".format(_train_set, _status)

                best_f1 = val_f_measure

                torch.save(model.state_dict(), mod_fp)
                print('Saved model to ' + mod_fp)

                val_counter = 0
            
            else:
                val_counter += 1

            train_loss.append(full_train_loss)
            val_loss.append(full_val_loss)
            train_f1.append(train_f_measure)
            val_f1.append(val_f_measure)

            if (val_counter >= 20):
                break
            
        mod_fp = "models/saved/cross_{}_{}.pth".format(_train_set, _status)

        if (_status == "pretrained"):
            test_mod = Down_CNN().cuda()
        else:
            test_mod = DS_CNN(pretext=True).cuda()

        state_dict = torch.load(mod_fp)
        test_mod.load_state_dict(state_dict)

        test_mod.eval()

        full_test_loss = 0.
        test_f_measure, test_cmlc, test_cmlt, test_amlc, test_amlt  , test_info_gain   = 0., 0., 0., 0., 0., 0.

        for batch_idx, wav in enumerate(test_wavs):
            with torch.no_grad():
                times = test_real_times[wav]

                if (_status == 'pretrained'):
                    vqt  = test_vqts[wav]
                    vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                    vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()     

                    msk = test_masks[wav]
                    msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                    output = model(vqt1, vqt2)
                    
                    loss = criterion(output, msk)

                else:
                    vqt = test_vqts[wav]
                    vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                    msk = test_masks[wav]
                    msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                    # Apply model to full batch
                    output = model(vqt)
                    
                    loss = criterion(output, msk)
                
                full_test_loss += loss.item()

                cpu_output = output.squeeze(0).cpu().detach().numpy()

                res = eval(cpu_output, times, threshold=threshold)
                test_f_measure += res[0]
                test_cmlc      += res[1]
                test_cmlt      += res[2]
                test_amlc      += res[3]
                test_amlt      += res[4]
                test_info_gain += res[5]

        full_test_loss = full_test_loss / (batch_idx + 1)
        test_f_measure = test_f_measure / (batch_idx + 1)
        test_cmlc      = test_cmlc      / (batch_idx + 1)
        test_cmlt      = test_cmlt      / (batch_idx + 1)
        test_amlc      = test_amlc      / (batch_idx + 1)
        test_amlt      = test_amlt      / (batch_idx + 1)
        test_info_gain = test_info_gain / (batch_idx + 1)

        print("\n-- Test Set --")

        print('\nMean test loss     is {:.3f}.'.format(full_test_loss))
        print('Mean beat F1-score is {:.3f}.'.format(test_f_measure))
        print('Mean beat CMLC     is {:.3f}.'.format(test_cmlc))
        print('Mean beat CMLT     is {:.3f}.'.format(test_cmlt))
        print('Mean beat AMLC     is {:.3f}.'.format(test_amlc))
        print('Mean beat AMLT     is {:.3f}.'.format(test_amlt))
        print('Mean beat InfoGain is {:.3f}.'.format(test_info_gain))

        f1.append(test_f_measure)
        cmlc.append(test_cmlc)
        cmlt.append(test_cmlt)
        amlc.append(test_amlc)
        amlt.append(test_amlt)
        ig.append(test_info_gain)

    f1   = np.asarray(f1)
    cmlc = np.asarray(cmlc)
    cmlt = np.asarray(cmlt)
    amlc = np.asarray(amlc)
    amlt = np.asarray(amlt)
    ig   = np.asarray(ig)

    print("\n8-fold CV results:")

    print('\nBeat F1-score is {:.3f} +- {:.3f}.'.format(np.mean(f1), np.std(f1)))
    print('Beat CMLC     is {:.3f} +- {:.3f}.'.format(np.mean(cmlc), np.std(cmlc)))
    print('Beat CMLT     is {:.3f} +- {:.3f}.'.format(np.mean(cmlt), np.std(cmlt)))
    print('Beat AMLC     is {:.3f} +- {:.3f}.'.format(np.mean(amlc), np.std(amlc)))
    print('Beat AMLT     is {:.3f} +- {:.3f}.'.format(np.mean(amlt), np.std(amlt)))
    print('Beat InfoGain is {:.3f} +- {:.3f}.'.format(np.mean(ig), np.std(ig)))

    return model
            

if __name__ == '__main__':
    # Load YAML file configuations 
    stream  = open("configuration/config.yaml", 'r')
    ymldict = yaml.safe_load(stream)

    _status = ymldict.get("cross_status")
    _train_set = ymldict.get("cross_train_set")

    print("Loading audio and pulses...")

    with open('GTZAN/wavs.pkl', 'rb') as handle:
        test_wavs = pickle.load(handle)

    with open('GTZAN/beat_pulses.pkl', 'rb') as handle:
        test_masks = pickle.load(handle)

    with open('GTZAN/real_beat_times.pkl', 'rb') as handle:
        test_real_times = pickle.load(handle)

    if (_status == 'pretrained'): 
        with open('GTZAN/vqts_spleeted.pkl', 'rb') as handle:
            test_vqts = pickle.load(handle)

    else:
        with open('GTZAN/vqts_original.pkl', 'rb') as handle:
            test_vqts = pickle.load(handle)

    if (_train_set == "hainsworth"):
        with open('Hainsworth/wavs.pkl', 'rb') as handle:
            train_wavs = pickle.load(handle)

        with open('Hainsworth/beat_pulses.pkl', 'rb') as handle:
            train_masks = pickle.load(handle)

        with open('Hainsworth/real_beat_times.pkl', 'rb') as handle:
            train_real_times = pickle.load(handle)

        if (_status == 'pretrained'):
            with open('Hainsworth/vqts_spleeted.pkl', 'rb') as handle:
                train_vqts = pickle.load(handle)

        else:
            with open('Hainsworth/vqts_original.pkl', 'rb') as handle:
                train_vqts = pickle.load(handle)

    elif (_train_set == "smc"):
        with open('SMC/smc_wavs.pkl', 'rb') as handle:
            train_wavs = pickle.load(handle)

        with open('SMC/smc_pulses.pkl', 'rb') as handle:
            train_masks = pickle.load(handle)

        with open('SMC/smc_real_times.pkl', 'rb') as handle:
            train_real_times = pickle.load(handle)

        if (_status == 'pretrained'):
            with open('SMC/smc_vqts_spleeted.pkl', 'rb') as handle:
                train_vqts = pickle.load(handle)
        
        else:
            with open('SMC/smc_vqts_original.pkl', 'rb') as handle:
                train_vqts = pickle.load(handle)  

    elif (_train_set == "ballroom"):
        with open('Ballroom/wavs.pkl', 'rb') as handle:
            train_wavs = pickle.load(handle)

        with open('Ballroom/beat_pulses.pkl', 'rb') as handle:
            train_masks = pickle.load(handle)

        with open('Ballroom/real_beat_times.pkl', 'rb') as handle:
            train_real_times = pickle.load(handle)

        if (_status == 'pretrained'):
            with open('Ballroom/vqts_spleeted.pkl', 'rb') as handle:
                train_vqts = pickle.load(handle)

        else:
            with open('Ballroom/vqts_original.pkl', 'rb') as handle:
                train_vqts = pickle.load(handle)
    
    else:
        print("YAML file for cross data set has a bug in experiment definition!")


    _ = train_model(train_wavs, train_vqts, train_masks, train_real_times, \
                    test_wavs , test_vqts , test_masks , test_real_times , \
                    ymldict)