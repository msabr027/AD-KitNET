# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:20:03 2019

@author: Mohamed Sabri
"""
# pip install opencv-python

import os
import time
import pandas as pd
import keras
from keras.preprocessing import image
from sklearn import preprocessing
import utils
#import os
#os.chdir(r"C:\Users\Mohamed Sabri\Desktop\AD_AI_Project\prod_file-exec")

if __name__ == "__main__":
    
    print('Main Starting...')
    parser = utils.parser_run_model()
    settings = vars(parser.parse_args())
    
    if settings["gpu"]==True:
        import cupy as np
        import KitNET_gpu as kit
        import utils_gpu as utils
    else:
        import numpy as np
        import KitNET as kit
        
    #data_main = pd.read_csv("./data/bank.csv")
    if settings["type"]=="num":
        
        print("loading data...")
        if settings["format"] =="csv":    
            data_main = pd.read_csv(settings["file"])
        if settings["format"] =="hdf":    
            data_main = pd.read_hdf(settings["file"])
        if settings["format"] =="excel":    
            data_main = pd.read_excel(settings["file"])
        if settings["format"] =="parquet":    
            data_main = pd.read_parquet(settings["file"])
        if settings["format"] =="json":    
            data_main = pd.read_json(settings["file"])
        d_cols = [col for col in data_main.columns if 'id' in col or 'index' in col or 'ID' in col or 'INDEXE' in col]
        data_main.drop(d_cols, inplace=True, axis=1)
        col = [c for c in data_main.columns]
        numclasses=[]
        for c in col:
           numclasses.append(len(np.unique(data_main[[c]])))
        threshold1=len(data_main)*0.2
        threshold2=2
        dummy_variables = list(np.array(col)[np.array(numclasses)==threshold2])
        collectdf1=[]
        for name in dummy_variables:
            if data_main[name].dtype == np.object:
                df1 = pd.get_dummies(data_main[name],prefix=name,drop_first=True,dummy_na=True)
                data_main.drop(name,axis=1,inplace=True)
                collectdf1=pd.concat([pd.DataFrame(collectdf1),df1],axis=1)
        categorical_variables = list(np.array(col)[(np.array(numclasses)<threshold1) & (np.array(numclasses)>threshold2)])
        collectdf2=[]
        for name2 in categorical_variables:
            df2 = pd.get_dummies(data_main[name2],prefix=name2,dummy_na=True)
            data_main.drop(name2,axis=1,inplace=True)
            collectdf2=pd.concat([pd.DataFrame(collectdf2),df2],axis=1)
        data_main.dropna(axis=1, how='all',inplace=True)
        X_scaled = preprocessing.scale(data_main)
        data_ready = pd.concat([pd.DataFrame(X_scaled),collectdf1,collectdf2],axis=1)
        
    if settings["type"]=="image":
        data_list = os.listdir("./data/images")
        train_image = []
        train_image = pd.DataFrame(train_image)
        for i in range(len(data_list)):
            img = image.load_img('./data/images/' + str(i+1) +  '.' + settings['format'], target_size=(settings["imgsize1"],settings["imgsize2"],1), grayscale=settings["gray"])
            img = image.img_to_array(img)
            img = img/255
            img = img.reshape(-1, img.shape[0]*img.shape[1]*img.shape[2])
            train_image = train_image.append(pd.DataFrame(img))
        data_ready = train_image
        
    epoch = settings["epoch"]
    maxAE = min(data_ready.shape[1],40) #maximum size for any autoencoder in the ensemble layer
    FMgrace = int(data_ready.shape[0])*epoch #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = int(data_ready.shape[0])*epoch #the number of instances used to train the anomaly detector (ensemble itself)    
    X = data_ready.values 
    K = kit.KitNET(X.shape[1],maxAE,FMgrace,ADgrace) # Build KitNET
    print("Running FMAEE:")
    start = time.time()
    # Here we process (train/execute) each individual observation.
    # In this way, X is essentially a stream, and each observation is discarded after performing process() method.
    for j in range(1,epoch*2+1):
        for i in range(X.shape[0]):
            K.process(X[i,]) #will train during the grace periods, then execute on all the rest.
            if i % X.shape[0] == 0:
                print(str(j) + " epoch")
    K.process(X[0,]) #will trigger saving the models
    stop = time.time()
    print("Training completed in: "+ str(round(stop - start)) +" seconds")
    
    RMSEs = np.zeros(X.shape[0]) # a place to save the scores
    for i in range(X.shape[0]):
        RMSEs[i] = K.execute(X[i,])
    if settings["sens"]=="low":
        threshold = np.mean(RMSEs) + np.std(RMSEs)
        level=1
    if settings["sens"]=="med":
        threshold = np.mean(RMSEs) + 2*np.std(RMSEs)
        level=2
    if settings["sens"]=="high":
        threshold = np.mean(RMSEs) + 3*np.std(RMSEs)
        level=3
        
    params = [threshold,level,np.mean(RMSEs),np.std(RMSEs)]
    np.save("./models/threshold.npy",params)
