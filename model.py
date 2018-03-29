# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:22:41 2018

@author: LIKS
"""

import tensorflow as tf
import numpy as np

class FastText:
    def __init__(self,content_len,title_len,Embedding,n_class):
        """
        Embedding:trained word2vec provided by zhihu
        """
        #placeholder for content, title, and label
        self.input_content=tf.placeholder(tf.int32,shape=[None,content_len],"input_content")
        self.input_title=tf.placeholder(tf.int32,shape=[None,title_len],"input_title")
        #为后期运算，Input_y应为浮点型
        self.input_y=tf.placeholder(tf.float32,shape=[None],"input_y")
        
        #embedding
        
        self.embed_W=tf.get_variable()("embed_W",shape=Embedding.shape,
                                    initializer=tf.constant_initializer(Embedding), trainable=True)
        self.input_content_embed=tf.nn.embedding_lookup(self.embed_W, self.input_content)#[None,content_len,embed_dim]
        self.input_content_mean=tf.reduce_mean(self.input_content_embed,1)#[None,embed_dim]
        self.input_title_embed=tf.nn.embedding_lookup(self.embed_W,self.input_title)#[None,title_len,embed_dim]
        self.input_title_mean=tf.reduce_mean(self.input_title_embed,1)#[None,embed_dim]
        
        #concatenate
        self.input=tf.concat([self.input_content_mean,self.input_title_mean],1)#[None,embed_dim*2]
        
        self.output_W=tf.get_variable("output_W",shape=[Embedding.shape[1]*2,n_class])
        self.output_b=tf.get_variable("output_b",shape=[n_class])
        self.outputs=tf.nn.xw_plus_b(self.input,self.output_W,self.output_b,name="outputs")#[None,n_class]
        
        self.loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs,labels=self.input_y,name="loss")
        

                
        
