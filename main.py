#!/usr/bin/env python3
import numpy as np
import sys
from Model import EEGNet
from load_data import load_data
from training import stage_training, standard_training
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import os
import shutil
from tabulate import tabulate
import gc
import tensorflow as tf

def main(args):
    bw = np.array([[args.frequency_cut_low,args.frequency_cut_high]])
    t_start      = np.int(250.0 * 2.5)
    t_end        = np.int(250.0 * 6) 
    t_sample  = 875
    n_channel = 22
    n_classes = 4

    callback = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    )]
    subjects = [1]#,2,3,4,5,6,7,8,9]
    Base_model = EEGNet(nb_classes = n_classes ,Chans = n_channel, Samples= t_sample)
    if args.cross_subject :
        [All_data_train,All_label_train,All_data_eval,All_label_eval] = load_data(subjects,args.path,bw,n_classes,t_start,t_end,t_sample)
        train_data,validation_data,train_lbl,validation_lbl = train_test_split(All_data_train,All_label_train)
        Base_model = stage_training(Base_model,train_data,train_lbl,
                    validation_data, validation_lbl,
                    callback,epochs = [args.epochs,args.fine_tune_epochs])

        Base_model.save_weights('./models/base_model')
        if args.subject == 0 :
            pred_EEGNet = Base_model.predict(All_data_eval)
            pred_EEGNet = np.argmax(pred_EEGNet,axis = 1)
            eval_label = np.argmax(All_label_eval,axis = 1)
            accuracy = accuracy_score(y_true=eval_label, y_pred=pred_EEGNet)
            output = [
                ["Model","EEGNet"],
                ["Stage training", 'Enable' if args.stage else 'Disable'],
                ["Training Epochs ", args.epochs],
                ["Accuracy ",accuracy]
            ]
            print(tabulate(output))
            sys.exit()  
    
    [All_data_train,All_label_train,All_data_eval,All_label_eval] = load_data([args.subject],args.path,bw,n_classes,t_start,t_end,t_sample)
    kf = KFold(n_splits=4)
    accuracies = []
    for iter in range(args.iterations):
        for train_index, valid_index in kf.split(All_data_train):
            gc.collect()
            train_data      =   All_data_train[train_index]
            validation_data =   All_data_train[valid_index]
            train_lbl       =   All_label_train[train_index]
            validation_lbl  =   All_label_train[valid_index]
            EEGNet_model = EEGNet(nb_classes = n_classes ,Chans = n_channel, Samples= t_sample)
            EEGNet_model.load_weights('./models/base_model')
            if args.stage :
                EEGNet_model = stage_training(EEGNet_model,train_data,train_lbl,
                            validation_data, validation_lbl,
                            callback,epochs = [args.epochs,args.fine_tune_epochs])
            else:
                EEGNet_model = standard_training(EEGNet_model,train_data,train_lbl,
                            validation_data, validation_lbl,
                            callback,epochs = [args.epochs,args.fine_tune_epochs])

            pred_EEGNet = EEGNet_model.predict(All_data_eval)
            pred_EEGNet = np.argmax(pred_EEGNet,axis = 1)
            #print(pred_EEGNet.shape)
            eval_label = np.argmax(All_label_eval,axis = 1)
            #print(eval_label.shape)
            accuracy = accuracy_score(y_true=eval_label, y_pred=pred_EEGNet)
            accuracies.append(accuracy)
            '''print('EEGNet acc : ',accuracy)
            print("=============================================================")'''
    
    output = [
        ["Model","EEGNet"],
        ["Stage training", 'Enable' if args.stage else 'Disable'],
        ["Cross Subjects", 'Enable' if args.cross_subject else 'Disable'],
        ["Training Epochs ", args.epochs],
        ["Training Iterations ", args.iterations],
        ["K_Fold ", 'Enable' if args.k_fold else 'Disable'],
        ["Accuracy ",np.mean(accuracies)]
    ]
    print(tabulate(output))
    if(not args.save_model):
        shutil.rmtree("./models")

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(description='Stage and Cross training strategy')
    parser.add_argument('--path', action="store",dest = "path", default="../BCI/dataset/Competition_iv_2a/",
                        help="path to the dataset folder see : bcni datasets fotmat")
    parser.add_argument('--patience', action="store", dest="patience", type = int ,default=1,
                    help="early stopping callback patience")
    parser.add_argument('--epochs'  , action="store", dest="epochs"  , default=1, type = int,
                    help="epochs model will be trained on")
    parser.add_argument('--frequency_cut_low' , action="store", dest="frequency_cut_low", type = float, default=4,
                    help="lower cut-off frequency in proprocessing")
    parser.add_argument('--frequency_cut_high', action="store", dest="frequency_cut_high", type = float, default=40,
                    help="higher cut-off frequency in proprocessing")
    parser.add_argument('--subject' , action="store", dest="subject", type = int, default= 0,
                    help="target subject")
    parser.add_argument('--k_fold', action="store", dest="k_fold", type = bool,default=True )
    parser.add_argument('--iterations' , action="store", dest="iterations", type = int, default=1)
    parser.add_argument('--fine_tune_epochs', action="store", dest="fine_tune_epochs", type = int,default=1 )
    parser.add_argument('--save_model', action="store", dest="save_model", type = int,default=1 ,
                    help="if true cross trained model wont be deleted after execution")
    
    parser.add_argument('--stage' , action="store_true", dest="stage", default = False,
                    help="if true stage training will be used instead of standard training")
    parser.add_argument('--cross_subject' , action="store_true", dest="cross_subject", default = False,
                    help="model will be pre-trained on all subjects of the data set")

    
    args = parser.parse_args()
    #print(args)
    main(args)
    sys.exit()
