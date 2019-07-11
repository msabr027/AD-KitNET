# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:21:03 2019

@author: Mohamed Sabri
"""

import os
import time
import pandas as pd
import numpy as np
import KitNET as kit
import utils
import keras
from keras.preprocessing import image
from sklearn import preprocessing
#import os
#os.chdir(r"C:\Users\Mohamed Sabri\Desktop\AD_AI_Project\prod_file-exec")

if __name__ == "__main__":
    
    print('Main Starting...')
    parser = utils.parser_run_model()
    settings = vars(parser.parse_args())
    
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
        
    XT = data_ready.values
    RMSE = np.zeros(XT.shape[0])
    
    epoch = settings["epoch"]
    maxAE = min(data_ready.shape[1],40) #maximum size for any autoencoder in the ensemble layer
    FMgrace = int(data_ready.shape[0])*epoch #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = int(data_ready.shape[0])*epoch #the number of instances used to train the anomaly detector (ensemble itself)   
    
    K = kit.KitNET(XT.shape[1],maxAE,FMgrace,ADgrace,saved=True) # Build KitNET

    for i in range(len(XT)):
        RMSE[i] = K.execute(XT[i,])
    
    thres = np.load("./models/threshold.npy")
    
    fault = (RMSE>=thres[0])
    sub = []
    sub = pd.DataFrame(sub)
    sub["values"] = (RMSE-thres[2])/(thres[1]*thres[3])
    sub["values"][sub["values"]<0]=0
    sub["values"][sub["values"]>1]=1
    sub["fault"] = fault
    sub.to_csv("./results/output.csv", sep = ",",index=False)
