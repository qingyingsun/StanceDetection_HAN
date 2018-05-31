#! /usr/bin/env python
# -*- coding: utf-8 -*-
#coding=utf-8
from __future__ import division
import math
import numpy as np  
import operator
import cPickle as cp
import os

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.metrics import average_precision_score
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from prepare_data_attention_allfactor_ownreason import get_data_ebeddinglayer
from keras.models import Sequential, Model
from keras.layers.core import *
from keras.optimizers import *
from keras.layers import *
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras import backend as K


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# set GPU memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


ISOTIMEFORMAT='%Y-%m-%d %X'
nb_classes=2
BATCH_SIZE=32
EMBEDDING_DIM=300
MAX_SEQUENCE_LENGTH=200
lstm_output_dim=300
nb_epoch = 15
hidden_dim = 128
UNIT_LENGTH=4


def calF1(best_predict_S, testLabels):
    true_l=[0]*nb_classes
    pred_l=[0]*nb_classes
    acl_l=[0]*nb_classes

    for pred_s, acl_s in zip(best_predict_S,testLabels):
        pred_l[pred_s]+=1
        acl_l[acl_s]+=1

        if pred_s==acl_s:
            true_l[acl_s]+=1

    precision=[0]*nb_classes
    recall=[0]*nb_classes
    f1=[0]*nb_classes

    for i in range(nb_classes):
        if pred_l[i]==0:
            precision[i]=0
        else:
            precision[i]=true_l[i]/(pred_l[i])
        if acl_l[i]==0:
            recall[i]=0
        else:
            recall[i]=true_l[i]/(acl_l[i])
        if(precision[i]+recall[i]<0.0001):
            f1[i]=0
        else:
            f1[i]=2*precision[i]*recall[i]/(precision[i]+recall[i])
    print((true_l[0]+true_l[1])/len(testLabels))
    for i in range(2):
        if i==0:
            print('the precision of against:%.6f'% precision[i])
            print('the recall of against:%.6f'% recall[i])
            print('the f1 of against:%.6f'% f1[i])
        elif i==1:
            print('the precision of favor:%.6f'% precision[i])
            print('the recall of favor:%.6f'% recall[i])
            print('the f1 of favor:%.6f'% f1[i])
    print 'average f1:',(f1[0]+f1[1])/2
    
    return f1

def returnLabel(predict_list):
    class_num_list = []
    for item in predict_list.tolist():
        max_score = 0
        class_num = 0
        for i, each_score in enumerate(item):
            if max_score <= each_score:
                max_score = each_score
                class_num = i
        class_num_list.append(class_num)
    return class_num_list

def ensemble_epoches(predictions_list):
    new_predictions_list=[]
    tt=0
    for probsList_oneS_allEpoches in zip(*predictions_list):
        sum_probList=[0]*len(probsList_oneS_allEpoches[0])
        for probs_oneEpoch in probsList_oneS_allEpoches:
            if tt==10:
                print probs_oneEpoch
            sum_probList=map(lambda x,y:x+y,probs_oneEpoch,sum_probList)
        if tt==10:
            print sum_probList
        new_predictions_list.append(sum_probList)
        tt+=1
    new_predictions_list=np.array(new_predictions_list)
    return new_predictions_list

def attention_model(input_matrix1,input_matrix2):
    M_matrix=input_matrix1
    dense_a = TimeDistributed(Dense(1,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform'))(M_matrix)
    print 'dense_a:',
    print 'dense_a', dense_a._keras_shape  #
    dense_a = Lambda(lambda x: x, output_shape=lambda s:s)(dense_a)
    dense_a=Reshape((-1,MAX_SEQUENCE_LENGTH))(dense_a)
    print 'dense_a after reshape', dense_a._keras_shape  #
    dense_a=Activation(activation="softmax")(dense_a)
    print 'dense_a after softmax', dense_a._keras_shape  #

    dense_r = Dot(axes=[2,1])([dense_a,input_matrix2])
    print 'dense_r', dense_r._keras_shape  #
    attention_represention = Flatten()(dense_r)
    print 'attention_represention', attention_represention._keras_shape  #
    return attention_represention

def attention_model_sentence(input_matrix1,input_matrix2):
    M_matrix=input_matrix1
    dense_a = TimeDistributed(Dense(1,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform'))(M_matrix)
    print 'dense_a:',
    print 'dense_a', dense_a._keras_shape  #
    dense_a = Lambda(lambda x: x, output_shape=lambda s:s)(dense_a)
    dense_a=Reshape((-1,UNIT_LENGTH))(dense_a)
    print 'dense_a after reshape', dense_a._keras_shape  #
    dense_a=Activation(activation="softmax")(dense_a)
    print 'dense_a after softmax', dense_a._keras_shape  #

    dense_r = Dot(axes=[2,1])([dense_a,input_matrix2])
    print 'dense_r', dense_r._keras_shape  #
    attention_represention = Flatten()(dense_r)
    print 'attention_represention', attention_represention._keras_shape  #
    return attention_represention
	
	
def lstm_attention_togeterfactor_model(train1,test1,embedding_matrix1,train2,test2,embedding_matrix2,\
                        train3,test3,embedding_matrix3,train4,test4,embedding_matrix4,trainLabels,\
                        testLabels,train_factor,test_factor,embedding_factor,target,test):
    print 'model construction start...'
    inputs1=Input(shape = (MAX_SEQUENCE_LENGTH,), name = 'words_input')
    inputs2=Input(shape = (MAX_SEQUENCE_LENGTH,), name = 'relation_word_input')
    inputs3=Input(shape = (MAX_SEQUENCE_LENGTH,), name = 'reason_input')
    inputs4=Input(shape = (MAX_SEQUENCE_LENGTH,), name = 'sentiment_input')
    inputs_factor=Input(shape = (UNIT_LENGTH,), name = 'factor_input')

    print 'inputs1',inputs1._keras_shape, 'inputs2',inputs2._keras_shape,'inputs3',inputs3._keras_shape,\
        'inputs4', inputs4._keras_shape,'inputs_factor',inputs_factor._keras_shape

    num_words1=len(embedding_matrix1)
    num_words2=len(embedding_matrix2)
    num_words3=len(embedding_matrix3)
    num_words4=len(embedding_matrix4)
    num_words_factor=len(embedding_factor)

    print 'num_words1=',num_words1
    print 'num_words2=',num_words2
    print 'num_words3=',num_words3
    print 'num_words4=',num_words4

    inputs1_embedding = Embedding(num_words1,
                                  EMBEDDING_DIM,
                                  weights=[embedding_matrix1],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  mask_zero=True
                                  )(inputs1)
    inputs2_embedding = Embedding(num_words2,
                                  EMBEDDING_DIM,
                                  weights=[embedding_matrix2],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  mask_zero=True
                                  )(inputs2)
    inputs3_embedding = Embedding(num_words3,
                                  EMBEDDING_DIM,
                                  weights=[embedding_matrix3],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  mask_zero=True
                                  )(inputs3)
    inputs4_embedding = Embedding(num_words4,
                                  EMBEDDING_DIM,
                                  weights=[embedding_matrix4],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  mask_zero=True
                                  )(inputs4)
    inputsfactor_embedding = Embedding(num_words_factor,
                                  EMBEDDING_DIM,
                                  weights=[embedding_factor],
                                  input_length=UNIT_LENGTH
                                  #trainable=False
                                  )(inputs_factor)			

    lstm1 = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                 recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                 unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs1_embedding)

    lstm1_Matrix = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                        recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                        return_sequences=True, unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs1_embedding)

    lstm2 = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                        recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                        unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs2_embedding)

    lstm2_Matrix = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                        recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                        return_sequences=True, unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs2_embedding)

    lstm3 = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                        recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                        unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs3_embedding)

    lstm3_Matrix = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                        recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                        return_sequences=True, unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs3_embedding)
    
    lstm4 = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                        recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                        unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs4_embedding)

    lstm4_Matrix = LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform",
                        recurrent_initializer="uniform", kernel_constraint=maxnorm(3), activation="tanh",
                        return_sequences=True, unit_forget_bias=True, dropout=0.2, recurrent_activation="sigmoid")(inputs4_embedding)

    print 'lstm1', lstm1._keras_shape, 'lstm2', lstm2._keras_shape, 'lstm3', lstm3._keras_shape, 'lstm4', lstm4._keras_shape

    unigram_Matrix = TimeDistributed(Dense(EMBEDDING_DIM,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform'))(lstm1_Matrix)
    relation_matrix = TimeDistributed(Dense(EMBEDDING_DIM,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform'))(lstm2_Matrix)
    reason_matrix = TimeDistributed(Dense(EMBEDDING_DIM,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform'))(lstm3_Matrix)
    sentiment_matrix = TimeDistributed(Dense(EMBEDDING_DIM,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform'))(lstm4_Matrix)
    
    attention_representation_dep=attention_model(relation_matrix,unigram_Matrix)
    attention_representation_reason=attention_model(reason_matrix,unigram_Matrix)
    attention_representation1_sentiment=attention_model(sentiment_matrix,unigram_Matrix)
    
    merged_layer = Concatenate(axis=-1)([lstm1,attention_representation_dep,attention_representation_reason,attention_representation1_sentiment])
    print 'merged_layer', merged_layer._keras_shape  
    merged_layer=Reshape((-1,lstm_output_dim))(merged_layer)
    print 'merged_layer', merged_layer._keras_shape    
    
    bilstm_matrix=Bidirectional(LSTM(lstm_output_dim,recurrent_dropout=0.2, kernel_initializer="uniform", recurrent_initializer="uniform", \
                                     kernel_constraint=maxnorm(3), activation='tanh',return_sequences=True, unit_forget_bias=True, \
                                     dropout=0.2, recurrent_activation="sigmoid"), merge_mode='concat')(merged_layer)
    

    bilstm_matrix = TimeDistributed(Dense(EMBEDDING_DIM*2,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform'))(bilstm_matrix)
    
    print 'bilstm_matrix', bilstm_matrix._keras_shape
    	
    attention_representation_sentence=attention_model_sentence(inputsfactor_embedding,bilstm_matrix)
 
   
    dense_h = Dense(200,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform', activation = 'relu')(attention_representation_sentence)
    dense_h=Dropout(0.25)(dense_h)
    dense_h = Dense(300,kernel_constraint=maxnorm(3),kernel_initializer = 'glorot_uniform', activation = 'relu')(dense_h)
    dense_h = Dense(300,kernel_constraint=maxnorm(3), kernel_regularizer=l2(0.01),kernel_initializer = 'glorot_uniform', activation = 'relu')(dense_h)
    dense_h=Dropout(0.25)(dense_h)
    dense_h = Dense(300,kernel_constraint=maxnorm(3), kernel_regularizer=l2(0.01),kernel_initializer = 'glorot_uniform', activation = 'relu')(dense_h)
    dense_h=Dropout(0.25)(dense_h)
    

    predictions = Dense(2, activation = 'softmax')(dense_h)
    model=Model(outputs=[predictions],inputs=[inputs1,inputs2,inputs3,inputs4,inputs_factor])
    optmr = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = optmr, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()    
    
    prediction_allEpoches=[]
    for i in range(1, nb_epoch + 1):
        print '%dth epoch...' % i
        hist = model.fit([train1, train2,train3,train4,train_factor], trainLabels, batch_size = BATCH_SIZE, epochs = 1, verbose = 2,validation_split=0.1, shuffle = True)
        loss = hist.history['val_loss'][-1]
        acc = hist.history['val_acc'][-1]
        predict_S = model.predict([test1, test2,test3,test4,test_factor], batch_size = 32, verbose = 2)
        prediction_allEpoches.append([acc,loss,predict_S])
        predict_class = returnLabel(predict_S)
                

    predict_output=open("Inner_attention_predictions_of_all_epoches.pkl","w")
    cp.dump((prediction_allEpoches,testLabels),predict_output)
    predict_output.close()
    
    sort1prediction_allEpoches=sorted(prediction_allEpoches,key=operator.itemgetter(0),reverse=True)
    sort2prediction_allEpoches=sorted(sort1prediction_allEpoches,key=operator.itemgetter(1))
        
    
    prediction_allEpoches=sort2prediction_allEpoches[:5]

    prediction_allEpoches= [item[2] for item in prediction_allEpoches]
    predict_ensemble=ensemble_epoches(prediction_allEpoches)
    predict_class_ensemble = returnLabel(predict_ensemble)

    print "ensemble epoch results:"
    f1=calF1(predict_class_ensemble, testLabels)


def run_multifeatures(target):
    test=get_data_ebeddinglayer(target)
    print 'loading input model ...start...'

    input_cp=np.load("Stance_attention_allfactors_3D_%s.npz"%(target),"rb")
    train1_data,test1_data,train2_data,test2_data,train3_data,test3_data,\
    train4_data,test4_data,train_factor_data,test_factor_data,train_stance_label,\
    test_stance_label,embedding_matrix1,embedding_matrix2,embedding_matrix3,embedding_matrix4,\
    embedding_matrix_factor=input_cp["train1_data"],input_cp["test1_data"], \
                            input_cp["train2_data"],input_cp["test2_data"],\
                            input_cp["train3_data"],input_cp["test3_data"], \
                            input_cp["train4_data"],input_cp["test4_data"],\
                            input_cp["train_factor_data"], input_cp["test_factor_data"],\
                            input_cp["train_stance_label"], \
                            input_cp["test_stance_label"],input_cp["embedding_matrix1"],input_cp["embedding_matrix2"],\
                            input_cp["embedding_matrix3"],input_cp["embedding_matrix4"],input_cp["embedding_matrix_factor"]
    
    lstm_attention_togeterfactor_model(train1_data,test1_data,embedding_matrix1,train2_data,test2_data,embedding_matrix2,\
                        train3_data,test3_data,embedding_matrix3,train4_data,test4_data,embedding_matrix4,train_stance_label,\
                        test_stance_label,train_factor_data,test_factor_data,embedding_matrix_factor,target,test)
    
       

if __name__ == '__main__':
    run_multifeatures('abortion')
