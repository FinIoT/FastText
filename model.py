# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:22:41 2018

@author: LIKS
"""
print("modle module started...")
import tensorflow as tf
import numpy as np

class FastText:
    def __init__(self,content_len,title_len,Embedding,):
        """
        Embedding:trained word2vec provided by zhihu
        """
        #placeholder for content, title, and label
        self.input_content=tf.placeholder(tf.int32,shape=[None,content_len],"input_content")
        self.input_title=tf.placeholder(tf.int32,shape=[None,title_len],"input_title")
        self.input_y=tf.placeholder(tf.int32,shape=[None],"input_y")
        
        
        #get_variable自身会初始化glorot_random_uniform
        self.embed_W=tf.get_variable("embed_W",shape=[vocab_size,embed_dim])
        self.input_x_embed=tf.nn.embedding_lookup(self.embed_W, self.input_x)#[None,sentence_len,embed_dim]
        self.output_W=tf.get_variable("output_W",shape=[embed_dim,vocab_size])
        self.output_b=tf.get_variable("output_b",shape=[vocab_size])
        

                
        
