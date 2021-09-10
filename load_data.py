import numpy as np
import filters
import scipy as sp
from get_data import get_data
import tensorflow as tf

def load_data(subjects,data_path,bw,n_classes,t_start,t_end,t_sample = 875):
    All_data_train  = np.zeros((0,22,t_sample,1))
    All_label_train = np.zeros((0,))
    All_data_eval   = np.zeros((0,22,t_sample,1))
    All_label_eval  = np.zeros((0,))
    filter_coef = filters.load_filterbank(bw, 250 ,order=4 ,max_freq=40,ftype = 'butter')
    
    for subject in subjects :
        train_data,train_label  = get_data(subject,True,data_path)
        eval_data, eval_label   = get_data(subject,False,data_path)
        
        train_label -= 1
        eval_label  -= 1

        ## filter data using butterworth
        train_data_filtered = np.zeros((train_data.shape[0],train_data.shape[1],t_end - t_start))
        for i_trial in range(train_data.shape[0]):
            train_data_filtered[i_trial] = filters.butter_fir_filter(train_data[i_trial,:,:], filter_coef[0])[:,t_start:t_end]

        train_data_filtered = sp.signal.resample(train_data_filtered,t_sample,axis=2)
        train_data_filtered = train_data_filtered.reshape(np.hstack((train_data_filtered.shape,1)))
        All_data_train      = np.vstack((All_data_train, train_data_filtered ))
        All_label_train     = np.hstack((All_label_train,train_label))

        eval_data_filtered  = np.zeros((eval_data.shape[0],eval_data.shape[1],t_end - t_start))
        for i_trial in range(eval_data.shape[0]):
            eval_data_filtered[i_trial] = filters.butter_fir_filter(eval_data[i_trial,:,:], filter_coef[0])[:,t_start:t_end]

        eval_data_filtered  = sp.signal.resample(eval_data_filtered,t_sample,axis=2)
        eval_data_filtered  = eval_data_filtered.reshape(np.hstack((eval_data_filtered.shape,1)))
        All_data_eval       = np.vstack((All_data_eval, eval_data_filtered ))
        All_label_eval      = np.hstack((All_label_eval,eval_label))
    All_label_train = tf.keras.utils.to_categorical(All_label_train).reshape((All_label_train.shape[0],n_classes,1))
    All_label_eval  = tf.keras.utils.to_categorical(All_label_eval ).reshape((All_label_eval.shape[0],n_classes,1))
    return  [All_data_train,All_label_train,All_data_eval,All_label_eval]