#coding:utf-8

from __future__ import division
import gensim
import random
import time
import math
import numpy as np
import operator
import os
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from numpy import *

MAX_NB_WORDS=5000
EMBEDDING_DIM=300
MAX_SEQUENCE_LENGTH=200
UNIT_LENGTH=4


class Document:
    def __init__(self,id,postid,words,text,stance,reasons,depList):
        self.stance=stance
        self.words=words
        self.postid=postid
        self.id=id
        self.text=text
        self.reasons=reasons
        self.depList=depList

def readidsets(path,target):
    newpath=path+target+'_stance.txt'
    f=open(newpath,'r')
    idsets=[]
    for line in f:
        line=line.replace('\r\n','')
        p=line.split('\t')
        postid = p[1].strip()
        idsets.append(postid)
    f.close()
    return idsets

def readfenju(path,target):
    idsets=readidsets(path,target)
    idline=[None]*len(idsets)
    for i in range(len(idline)):
        idline[i]=['']*2
    newpath=path+target+'_newfenju.txt'
    f=open(newpath,'r')
    j=0
    i=0
    for line in f:
        if line!='' and line!='\r\n':
            i+=1
        elif line=='\r\n':
            idline[j][0]=idsets[j]
            idline[j][1]=str(i)
            i=0
            j+=1
    f.close()
    return idline

def readdeplist(path,target):
    deplist={}
    taglist={}
    idline=readfenju(path,target)
    newpath=path+target+'_dp_newfenju_result.txt'
    f=open(newpath,'r')
    etemdp=''
    i=0
    j=0
    for line in f:
        if line!='' and line!='\r\n':
            etemdp+=line
        elif line=='\r\n':
            i+=1
            etemdp+='\t'
            if str(i)==idline[j][1]:
                deplist[idline[j][0]]=etemdp
                j+=1
                i=0
                etemdp=''
    newpath2=path+target+'_fencitag_newfenju_result.txt'
    f2=open(newpath2,'r')
    i=0
    j=0
    itemtag=''
    for line in f2:
        if line!='' and line!='\r\n':
            itemtag+=line
            i+=1
        elif line=='\r\n':
            itemtag+='\t'
            if str(i)==idline[j][1]:
                taglist[idline[j][0]]=itemtag
                j+=1
                i=0
                itemtag=''

    f.close()
    f2.close()
    return deplist,taglist

def readReason(path):
    reasonsets={}
    for fname in os.listdir(path):
        for line in open(os.path.join(path,fname)):
            if line!='':
                line=line.replace('\r\n','')		
                p=line.split('\t')
                id,postid,text,reaflag=p[0],p[1],p[2],p[3]		
                if reaflag=='1':
                    if postid not in reasonsets:
                        reasonsets[postid]=[]
                        reasonsets[postid].append(text)
			#print 'reason0',text
                    else:
                        reasonsets[postid].append(text)
			#print 'reason1',text

    return reasonsets

def readFromFile(path,target):
    documents=[]
    deplist,taglist=readdeplist('../backupdata/',target)
    newpath1=path+target+'_stance.txt'
    newpath2='../resultreason/'+target+'/'
    reasonsets=readReason(newpath2)
    documents=[]
    f = open(newpath1,'r')
    for line in f:
        line=line.replace('\r\n','')
        p=line.split('\t')
        wordsetsnew=[]
        id,postid,text,stanceflag =p[0].strip(),p[1].strip(),p[2].strip(),p[3].strip()
        reasons=''
        if postid in reasonsets:
            for item in reasonsets[postid]:
                reasons+=item+' '
        if line!='' :
            text=text.lower()
            wordsets=text.strip().split(' ')

            if stanceflag=='-1':
                stance=0
            elif stanceflag=='+1':
                stance=1
            for word in wordsets:
                if len(word)>0:
                    wordsetsnew.append(word)
            dl=deplist[postid]
            documents.append(Document(id,postid,wordsetsnew,text,stance,reasons,dl))

    f.close()
    return documents

def matching(depstr):
    k1=depstr.find('(')
    relation=depstr[0:k1]
    k4=depstr.rfind('-')
    k3=depstr.rfind(',',0,k4-2)
    k2=depstr.rfind('-',0,k3)
    preword=depstr[k1+1:k2].strip()
    laterword=depstr[k3+1:k4].strip()
    preindex=int(re.findall(r"\d+",depstr[k2+1:k3])[0])
    laterindex=int(re.findall(r"\d+",depstr[k4+1:-1])[0])
    return relation,preword,laterword,preindex,laterindex

def createVec_tokenize(train,test,subjecDic):
    train1_text=[];  train2_text=[]; train3_text=[]; train4_text=[]    
    test1_text=[]; test2_text=[]; test3_text=[]; test4_text=[]
    train_factor_text=[]; test_factor_text=[]
    factorset=['unigramfactor','dpwordfactor','reasonfactor','sentimentfactor']

    train_stance_label=[]    
    test_stance_label=[]
    for doc in train:
        train1_text.append(doc.text)
        train_stance_label.append(doc.stance)
        depitem=doc.depList.strip().split('\t')
        relation_word=''
        for j in range(len(depitem)):
            itemsets=depitem[j].strip().split('\n')
            for item in itemsets:
                relation,preword,laterword,preindex,laterindex=matching(item)
                if relation in ('nsubj','nn','dobj','mark','cc','det','acl','acl:relcl','amod','cop','iobj','xcomp','neg', \
                        'conj:but','conj:and','conj:or','advcl:on','advcl:at','advcl:in','advcl:for','advcl:with','advcl:to', \
                        'nmod:to','nmod:poss','nmod:by','nmod:as','nmod:for','nmod:of','nmod:at','nmod:about','nmod:like', \
                        'nmod:upon','nmod:on','nmod:in'):
                    flag=preword+'0'+laterword
                    relation_word+=flag+' '
        train2_text.append(relation_word)
	
        train3_text.append(doc.reasons)
        sentiwordlist=''
        for word in doc.words:
            if word in subjecDic:
                sentiwordlist+=word+' '
        train4_text.append(sentiwordlist)
        factorlist=''
        for item in factorset:
            factorlist+=item+' '
        train_factor_text.append(factorlist)
   
    for doc in test:
        test1_text.append(doc.text)
        test_stance_label.append(doc.stance)
        depitem=doc.depList.strip().split('\t')
        relation_word=''
        for j in range(len(depitem)):
            itemsets=depitem[j].strip().split('\n')
            for item in itemsets:
                relation,preword,laterword,preindex,laterindex=matching(item)
                
                if relation in ('nsubj','nn','dobj','mark','cc','det','acl','acl:relcl','amod','cop','iobj','xcomp','neg', \
                        'conj:but','conj:and','conj:or','advcl:on','advcl:at','advcl:in','advcl:for','advcl:with','advcl:to', \
                        'nmod:to','nmod:poss','nmod:by','nmod:as','nmod:for','nmod:of','nmod:at','nmod:about','nmod:like', \
                        'nmod:upon','nmod:on','nmod:in'):
                    flag=preword+'0'+laterword
                    relation_word+=flag+' '
        test2_text.append(relation_word)
        test3_text.append(doc.reasons)
        sentiwordlist=''
        for word in doc.words:
            if word in subjecDic:
                sentiwordlist+=word+' '
        test4_text.append(sentiwordlist)
        factorlist=''
        for item in factorset:
            factorlist+=item+' '
        test_factor_text.append(factorlist)

    texts=train1_text+test1_text+train3_text+test3_text+train4_text+test4_text
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences_train1 = tokenizer.texts_to_sequences(train1_text)    
    sequences_test1 = tokenizer.texts_to_sequences(test1_text)

    sequences_train3 = tokenizer.texts_to_sequences(train3_text)    
    sequences_test3 = tokenizer.texts_to_sequences(test3_text)

    sequences_train4 = tokenizer.texts_to_sequences(train4_text)    
    sequences_test4 = tokenizer.texts_to_sequences(test4_text)

    word_index = tokenizer.word_index
    print('Found word_index %s unique tokens.' % len(word_index))

    texts_relation=train2_text+test2_text
    tokenizer_relation = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer_relation.fit_on_texts(texts_relation)
    sequences_train2 = tokenizer_relation.texts_to_sequences(train2_text)    
    sequences_test2 = tokenizer_relation.texts_to_sequences(test2_text)

    word_relation_index = tokenizer_relation.word_index
    print('Found word_relation_index %s unique tokens.' % len(word_relation_index))

    texts_factor=train_factor_text
    tokenizer_factor = Tokenizer(num_words=4)
    tokenizer_factor.fit_on_texts(texts_factor)
    sequences_train_factor = tokenizer_factor.texts_to_sequences(train_factor_text)    
    sequences_test_factor = tokenizer_factor.texts_to_sequences(test_factor_text)

    word_factor_index = tokenizer_factor.word_index
    print('Found word_factor_index %s unique tokens.' % len(word_factor_index))

    train1_data = pad_sequences(sequences_train1, maxlen=MAX_SEQUENCE_LENGTH)    
    test1_data = pad_sequences(sequences_test1, maxlen=MAX_SEQUENCE_LENGTH)

    train2_data = pad_sequences(sequences_train2, maxlen=MAX_SEQUENCE_LENGTH)    
    test2_data = pad_sequences(sequences_test2, maxlen=MAX_SEQUENCE_LENGTH)

    train3_data = pad_sequences(sequences_train3, maxlen=MAX_SEQUENCE_LENGTH)    
    test3_data = pad_sequences(sequences_test3, maxlen=MAX_SEQUENCE_LENGTH)

    train4_data = pad_sequences(sequences_train4, maxlen=MAX_SEQUENCE_LENGTH)    
    test4_data = pad_sequences(sequences_test4, maxlen=MAX_SEQUENCE_LENGTH)

    train_factor_data = pad_sequences(sequences_train_factor, maxlen=UNIT_LENGTH)    
    test_factor_data = pad_sequences(sequences_test_factor, maxlen=UNIT_LENGTH)

    train_stance_label = np_utils.to_categorical(np.asarray(train_stance_label))
    

    print('Shape of train1 data tensor:', train1_data.shape)    
    print('Shape of test1 data tensor:', test1_data.shape)
    print('Shape of train2 data tensor:', train2_data.shape)    
    print('Shape of test2 data tensor:', test2_data.shape)
    print('Shape of train label tensor:', train_stance_label.shape)    
    print('Shape of factor train data tensor:', train_factor_data.shape)    
    print('Shape of factor test tensor:', test_factor_data.shape)

    return train1_data,test1_data,train2_data,test2_data,train3_data,test3_data,\
           train4_data,test4_data,train_factor_data,test_factor_data,train_stance_label,\
           test_stance_label,word_index,word_relation_index,word_factor_index

def get_test_data(test,my_model1,my_model2):
    test_data=[]
    test_stance_label=[]
    for doc in test:
        sen=[]
        for word in doc.segtexts:
            if word in my_model1:
                sen.append(list(my_model1[word]))
            elif word.encode('utf-8') in my_model2:
                sen.append(list(pad_sequences(my_model2[word.encode('utf-8')], maxlen=EMBEDDING_DIM)))
            else:
                sen.append(np.zeros(EMBEDDING_DIM).astype(np.float))
        test_data.append(sen)
        test_stance_label.append(doc.stance)
    test_data_padding=pad_sequences(test_data, maxlen=MAX_SEQUENCE_LENGTH)
    return test_data_padding,test_stance_label

def create_ebedding_weights(word_index,model1):
    embedding_index={}
    countin=0
    for word in word_index:
        try:
            embedding_index[word]=model1[word.encode('utf-8')]
	        countin+=1	    
        except:            
	        embedding_index[word]=np.random.rand(EMBEDDING_DIM).astype(np.float)*2-1
	    
    embedding_matrix = np.random.rand(len(word_index) + 1, EMBEDDING_DIM)*2-1    
    
    print 'word in model:',countin
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:            
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix)
    return embedding_matrix
	
def create_ebedding_weights_factor_v2(word_index):
    factorset=['unigramfactor','dpwordfactor','reasonfactor','sentimentfactor']
    embedding_index={}
    for word in word_index:
        embedding_index[word]=[np.random.uniform(0,1) for i in range(EMBEDDING_DIM)]	        
    
    embedding_matrix =[([np.random.uniform(0,1)]*EMBEDDING_DIM) for i in range(len(word_index) + 1)]

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector   
    return embedding_matrix	


def readfivefolddata(path,target):
    stancefoldset={}
    reasonfoldset={}
    for fname in os.listdir(path):
        p=fname.split('-')
        if p[0]==target:
            for line in open(os.path.join(path,fname)):
                if line!='':
                    line=line.replace('\n','')
                    idsets=line.split(' ')
                    if idsets[0] not in stancefoldset:
                        stancefoldset[idsets[0]]=int(p[1])
                    if len(idsets)==2 and idsets[0] not in reasonfoldset:
                        reasonfoldset[idsets[0]]=int(p[1])
    return stancefoldset,reasonfoldset

def get_fold_data(alltrain,target):
    stancefoldset,reasonfoldset=readfivefolddata('../folds/',target)
    document1=[];document2=[];document3=[];document4=[];document5=[]
    for document in alltrain:
        postid=document.postid
        if postid in stancefoldset:
            if stancefoldset[postid]==1:
                document1.append(document)
            elif stancefoldset[postid]==2:
                document2.append(document)
            elif stancefoldset[postid]==3:
                document3.append(document)
            elif stancefoldset[postid]==4:
                document4.append(document)
            elif stancefoldset[postid]==5:
                document5.append(document)
    return document1,document2,document3,document4,document5

def createSubjectivityDic():
    f1=open('../data/subjcluesLexicon.tff')
    subjecDic={}

    for line in f1:
        line=line.replace('\n','')
        segParts=line.split(' ')
        if segParts[2][6:] not in subjecDic:
            if segParts[5][14:]=='positive':
                subjecDic[segParts[2][6:]]='pos'
            elif segParts[5][14:]=='negative':
                subjecDic[segParts[2][6:]]='neg'
            elif segParts[5][14:]=='both':
                subjecDic[segParts[2][6:]]='bt'
            elif segParts[5][14:]=='neutral':
                subjecDic[segParts[2][6:]]='neut'

    f1.close()
    return subjecDic

def get_data_ebeddinglayer(targetname):
    alltrain=readFromFile('../data/',targetname)
    subjecDic=createSubjectivityDic()
    print('len(train):'+str(len(alltrain)))

    document1,document2,document3,document4,document5=get_fold_data(alltrain,targetname)
    trainall=document4+document1+document5+document2
    test=document3
   
    my_model1=gensim.models.Word2Vec.load('../ebedding/model_300.m')
    my_model2=gensim.models.Word2Vec.load('../ebedding/model_word_word_300.m')
    print('type of my_model:'+str(type(my_model1)))

    train1_data,test1_data,train2_data,test2_data,train3_data,test3_data,\
           train4_data,test4_data,train_factor_data,test_factor_data,train_stance_label,\
           test_stance_label,word_index,word_relation_index,word_factor_index = createVec_tokenize(trainall,test,subjecDic)
    embedding_matrix1=create_ebedding_weights(word_index,my_model1)
    embedding_matrix2=create_ebedding_weights(word_relation_index,my_model2)
    embedding_matrix3=create_ebedding_weights(word_index,my_model1)
    embedding_matrix4=create_ebedding_weights(word_index,my_model1)
    
    embedding_matrix_factor=create_ebedding_weights_factor_v2(word_factor_index)

    np.savez("Stance_attention_allfactors_3D_%s.npz"%(targetname),\
             train1_data=train1_data,test1_data=test1_data,embedding_matrix1=embedding_matrix1,\
             train2_data=train2_data,test2_data=test2_data,embedding_matrix2=embedding_matrix2,\
             train3_data=train3_data,test3_data=test3_data,embedding_matrix3=embedding_matrix3,\
             train4_data=train4_data,test4_data=test4_data,embedding_matrix4=embedding_matrix4,\
             train_factor_data=train_factor_data,test_factor_data=test_factor_data,\
             embedding_matrix_factor=embedding_matrix_factor,train_stance_label=train_stance_label,\
             test_stance_label=test_stance_label)

    return test