import torch
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from models.models       import DS_CNN, Down_CNN
from processing.evaluate import beat_tracking as eval

def train_model(wavs, vqts, masks, real_times, data_set, ymldict):
    """
    Function for training model on Ballroom data set.
    -- wavs       : list of files
    -- vqts       : spectrograms of audio
    -- masks      : pulse vectors
    -- real_times : list with real times in seconds
    -- data_set   : data set we are running our experiment on
    -- ymldict    : YAML parameters
    """
    # Load the experiment stuff:
    _status = ymldict.get("{}_status".format(data_set))
    _pre    = ymldict.get("{}_pre".format(data_set))
    _exp    = ymldict.get("{}_exp".format(data_set))
    _lr     = ymldict.get("{}_lr".format(data_set))
    _eval   = ymldict.get("{}_eval".format(data_set))

    threshold = False

    if (_eval == 'threshold'):
        threshold = True

    random.shuffle(wavs)

    cv_len = len(wavs) / 8

    split1  = wavs[0:round(cv_len)]
    split2  = wavs[round(cv_len):round(cv_len * 2)]
    split3  = wavs[round(cv_len * 2):round(cv_len * 3)]
    split4  = wavs[round(cv_len * 3):round(cv_len * 4)]
    split5  = wavs[round(cv_len * 4):round(cv_len * 5)]
    split6  = wavs[round(cv_len * 5):round(cv_len * 6)]
    split7  = wavs[round(cv_len * 6):round(cv_len * 7)]
    split8  = wavs[round(cv_len * 7):]

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
    
        if ((_status == "pretrained" and _pre == "finetune") or 
            (_status == "pretrained" and _pre == "frozen"  ) or 
            (_status != "pretrained")):

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

            test_indices  = splits[jj]

            random.shuffle(train_indices)

            val_indices   = train_indices[0:round(cv_len)]
            train_indices = train_indices[round(cv_len):]

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
                    times = real_times[wav]

                    if (_status == 'pretrained'):
                        vqt  = vqts[wav]
                        vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                        vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()

                        msk = masks[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        optimizer.zero_grad()
                        
                        # Apply model to full batch
                        output = model(vqt1, vqt2)
                        
                        loss = criterion(output, msk)
                        loss.backward()
                        optimizer.step()

                    else:
                        vqt = vqts[wav]
                        vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                        msk = masks[wav]
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

                print('\nMean training loss     is {:.3f}.'.format(full_train_loss))
                print('Mean training F1-score is {:.3f}.'.format(train_f_measure))

                model.eval()

                for batch_idx, wav in enumerate(val_indices):
                    with torch.no_grad():
                        times = real_times[wav]

                        if (_status == 'pretrained'):
                            vqt  = vqts[wav]
                            vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                            vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()     

                            msk = masks[wav]
                            msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                            output = model(vqt1, vqt2)
                            
                            loss = criterion(output, msk)

                        else:
                            vqt = vqts[wav]
                            vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                            msk = masks[wav]
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
                    mod_fp = "models/saved/{}_{}_{}.pth".format(data_set, _exp, _status)

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
                
            mod_fp = "models/saved/{}_{}_{}.pth".format(data_set, _exp, _status)

            if (_status == "pretrained"):
                test_mod = Down_CNN().cuda()
            else:
                test_mod = DS_CNN(pretext=True).cuda()

            state_dict = torch.load(mod_fp)
            test_mod.load_state_dict(state_dict)

            test_mod.eval()

            full_test_loss = 0.
            test_f_measure  , test_cmlc  , test_cmlt  , test_amlc  , test_amlt  , test_info_gain   = 0., 0., 0., 0., 0., 0.

            for batch_idx, wav in enumerate(test_indices):
                with torch.no_grad():
                    times = real_times[wav]

                    if (_status == 'pretrained'):
                        vqt  = vqts[wav]
                        vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                        vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()     

                        msk = masks[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        output = model(vqt1, vqt2)
                        
                        loss = criterion(output, msk)

                    else:
                        vqt = vqts[wav]
                        vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                        msk = masks[wav]
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
            print('Mean test F1-score is {:.3f}.'.format(test_f_measure))
            print('Mean test CMLC     is {:.3f}.'.format(test_cmlc))
            print('Mean test CMLT     is {:.3f}.'.format(test_cmlt))
            print('Mean test AMLC     is {:.3f}.'.format(test_amlc))
            print('Mean test AMLT     is {:.3f}.'.format(test_amlt))
            print('Mean test InfoGain is {:.3f}.'.format(test_info_gain))
            
            f1.append(test_f_measure)
            cmlc.append(test_cmlc)
            cmlt.append(test_cmlt)
            amlc.append(test_amlc)
            amlt.append(test_amlt)
            ig.append(test_info_gain)

            if (jj == 0):
                pathlib.Path('figures/{}/{}'.format(data_set, _exp)).mkdir(parents=True, exist_ok=True) 

            plt.figure()
            plt.plot(train_loss)
            plt.plot(val_loss)
            plt.legend(['Train', 'Validation'])
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig("figures/{}/{}/{}_loss_{}.pdf".format(data_set, _exp, _status, jj), dpi=300, format='pdf')

            plt.figure()
            plt.plot(train_f1)
            plt.plot(val_f1)
            plt.legend(['Train', 'Validation'])
            plt.ylim([0, 1])
            plt.xlabel("Epochs")
            plt.ylabel("F1-score")
            plt.savefig("figures/{}/{}/{}_f1_{}.pdf".format(data_set, _exp, _status, jj), dpi=300, format='pdf')

        elif (_status == "pretrained" and _pre == "validation"):
            model.eval()

            full_test_loss = []
            test_f_measure  , test_cmlc  , test_cmlt  , test_amlc  , test_amlt  , test_info_gain   = [], [], [], [], [], []

            for batch_idx, wav in enumerate(wavs):
                with torch.no_grad():
                    times = real_times[wav]

                    if (_status == 'pretrained'):
                            vqt  = vqts[wav]
                            vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                            vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()     

                            msk = masks[wav]
                            msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                            output = model(vqt1, vqt2)
                            
                            loss = criterion(output, msk)

                    else:
                        vqt = vqts[wav]
                        vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                        msk = masks[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        # Apply model to full batch
                        output = model(vqt)
                        
                        loss = criterion(output, msk)
                    
                    full_test_loss.append(loss.item())

                    cpu_output = output.squeeze(0).cpu().detach().numpy()

                    res = eval(cpu_output, times, threshold=threshold)
                    test_f_measure.append(res[0])
                    test_cmlc.append(res[1])
                    test_cmlt.append(res[2])
                    test_amlc.append(res[3])
                    test_amlt.append(res[4])
                    test_info_gain.append(res[5])

            print("\n-- Full Set --")

            print('Mean loss     is {:.3f} +- {:.3f}.'.format(np.mean(full_test_loss), np.std(full_test_loss)))
            print('Mean F1-score is {:.3f} +- {:.3f}.'.format(np.mean(test_f_measure), np.std(test_f_measure)))
            print('Mean CMLC     is {:.3f} +- {:.3f}.'.format(np.mean(test_cmlc), np.std(test_cmlc)))
            print('Mean CMLT     is {:.3f} +- {:.3f}.'.format(np.mean(test_cmlt), np.std(test_cmlt)))
            print('Mean AMLC     is {:.3f} +- {:.3f}.'.format(np.mean(test_amlc), np.std(test_amlc)))
            print('Mean AMLT     is {:.3f} +- {:.3f}.'.format(np.mean(test_amlt), np.std(test_amlt)))
            print('Mean InfoGain is {:.3f} +- {:.3f}.'.format(np.mean(test_info_gain), np.std(test_info_gain)))

            break

        else:
            raise ValueError("Problem with configuration file experiment arguments: {} and {}.".format(_status, _pre))

    if (f1 != []):
        f1   = np.asarray(f1)
        cmlc = np.asarray(cmlc)
        cmlt = np.asarray(cmlt)
        amlc = np.asarray(amlc)
        amlt = np.asarray(amlt)
        ig   = np.asarray(ig)

        print("\n8-fold CV results:")
        print('\nF1-score is {:.3f} +- {:.3f}.'.format(np.mean(f1), np.std(f1)))
        print('CMLC     is {:.3f} +- {:.3f}.'.format(np.mean(cmlc), np.std(cmlc)))
        print('CMLT     is {:.3f} +- {:.3f}.'.format(np.mean(cmlt), np.std(cmlt)))
        print('AMLC     is {:.3f} +- {:.3f}.'.format(np.mean(amlc), np.std(amlc)))
        print('AMLT     is {:.3f} +- {:.3f}.'.format(np.mean(amlt), np.std(amlt)))
        print('InfoGain is {:.3f} +- {:.3f}.'.format(np.mean(ig), np.std(ig)))

    return model