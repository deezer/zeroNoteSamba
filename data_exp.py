import torch
import random
import numpy as np

from tqdm import trange
from models.models import DS_CNN, Down_CNN
from processing.evaluate import beat_tracking as eval


def train_model(wavs, vqts, beat_pulse, real_beat_times, data_set, ymldict):
    """
    Function for training model on GTZAN data set.
    -- wavs            : list of wav files (dictionary keys)
    -- vqts            : spectrograms of audio
    -- beat_pulse      : beat tracking pulse vectors
    -- real_beat_times : list with real beat tracking times in seconds
    -- data_set        : data set we are running our experiment on
    -- ymldict         : YAML parameters
    """
    # Load the experiment stuff:
    _status = ymldict.get("{}_status".format(data_set))
    _pre    = ymldict.get("{}_pre".format(data_set))
    _exp    = ymldict.get("{}_exp".format(data_set))
    _lr     = ymldict.get("{}_lr".format(data_set))

    random.Random(16).shuffle(wavs)

    cv_len = len(wavs) / 8

    split         = wavs[0:round(cv_len * 6)]
    val_indices   = wavs[round(cv_len * 6):round(cv_len * 7)]
    test_indices  = wavs[round(cv_len * 7):]

    split_len = len(split)

    percentages = [round(split_len * 0.01), round(split_len * 0.02), round(split_len * 0.05), round(split_len * 0.1), 
                   round(split_len * 0.2) , round(split_len * 0.5) , round(split_len * 0.75)]

    for perc in percentages:

        f1   = []
        cmlc = []
        cmlt = []
        amlc = []
        amlt = []
        ig   = []

        print("\nTrain set percentage is {}.".format(perc))

        for jj in range(10):

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

            random.Random(16).shuffle(split)

            train_indices   = split[0:perc]

            best_f1   = 0.

            # Train model
            for _ in trange(500):
                full_train_loss = 0.
                full_val_loss   = 0.
                train_f_measure, train_cmlc, train_cmlt, train_amlc, train_amlt, train_info_gain = 0., 0., 0., 0., 0., 0.
                val_f_measure  , val_cmlc  , val_cmlt  , val_amlc  , val_amlt  , val_info_gain   = 0., 0., 0., 0., 0., 0.

                model.train()
                
                for batch_idx, wav in enumerate(train_indices):
                    times = real_beat_times[wav]

                    if (_status == 'pretrained'):  
                        vqt  = vqts[wav]
                        vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                        vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()

                        msk = beat_pulse[wav]
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

                        msk = beat_pulse[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        optimizer.zero_grad()

                        # Apply model to full batch
                        output = model(vqt)
                        
                        loss = criterion(output, msk)
                        loss.backward()
                        optimizer.step()
                    
                    full_train_loss += loss.item()

                    cpu_output = output.squeeze(0).cpu().detach().numpy()

                    res = eval(cpu_output, times)
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

                model.eval()

                for batch_idx, wav in enumerate(val_indices):
                    with torch.no_grad():
                        times = real_beat_times[wav]

                        if (_status == 'pretrained'):
                            vqt  = vqts[wav]
                            vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                            vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()     

                            msk = beat_pulse[wav]
                            msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                            output = model(vqt1, vqt2)
                            
                            loss = criterion(output, msk)
                        
                        else:
                            vqt = vqts[wav]
                            vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                            msk = beat_pulse[wav]
                            msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                            # Apply model to full batch
                            output = model(vqt)
                            
                            loss = criterion(output, msk)
                        
                        full_val_loss += loss.item()

                        cpu_output = output.squeeze(0).cpu().detach().numpy()

                        res = eval(cpu_output, times)
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

                if (val_f_measure > best_f1):
                    mod_fp = "models/saved/{}_{}_{}.pth".format(data_set, _exp, _status)
                    best_f1 = val_f_measure
                    torch.save(model.state_dict(), mod_fp)
                    val_counter = 0
                
                else:
                    val_counter += 1

                train_loss.append(full_train_loss)
                val_loss.append(full_val_loss)
                train_f1.append(train_f_measure)
                val_f1.append(val_f_measure)

                if (val_counter >= 20):
                    break
                
            print("\nBest validation F1-score is {}.".format(best_f1))

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
                    times = real_beat_times[wav]

                    if (_status == 'pretrained'):
                        vqt  = vqts[wav]
                        vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                        vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()     

                        msk = beat_pulse[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        output = model(vqt1, vqt2)
                        
                        loss = criterion(output, msk)

                    else:
                        vqt = vqts[wav]
                        vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                        msk = beat_pulse[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        # Apply model to full batch
                        output = model(vqt)
                        
                        loss = criterion(output, msk)
                    
                    full_test_loss += loss.item()

                    cpu_output = output.squeeze(0).cpu().detach().numpy()

                    res = eval(cpu_output, times)
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

            print("\n-- Test Set {} --".format(jj))

            print('\nMean test loss     is {:.3f}.'.format(full_test_loss))
            print('Mean beat F1-score is {:.3f}.'.format(test_f_measure))
            
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