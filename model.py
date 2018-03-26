# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:22:41 2018

@author: LIKS
"""
print("modle module started...")
import tensorflow as tf
import numpy as np

class FastText:
    def __init__(self,sentence_len,vocab_size,embed_dim,model="train"):
        #self.sentence_len=sentence_len
        #placeholder for X, and Y; x should be fixed length
        self.input_x=tf.placeholder(tf.int32,shape=[None,sentence_len],"input_x")
        self.input_y=tf.placeholder(tf.int32,shape=[None],"input_y")
        #get_variable自身会初始化glorot_random_uniform
        self.embed_W=tf.get_variable("embed_W",shape=[vocab_size,embed_dim])
        self.input_x_embed=tf.nn.embedding_lookup(self.embed_W, self.input_x)#[None,sentence_len,embed_dim]
        self.output_W=tf.get_variable("output_W",shape=[embed_dim,vocab_size])
        self.output_b=tf.get_variable("output_b",shape=[vocab_size])
        self.nce_loss
    def loss():
        if model="train":
            self.loss=
                
        
