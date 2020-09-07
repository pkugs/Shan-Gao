import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.layers as tcl
import time
from speech_data import data_reader
import config
import matplotlib.pyplot as plt
import os
import logging
from numpy import *

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool(x,m,n):
	return tf.nn.max_pool(x,ksize=[1,m,n,1],strides=[1,m,n,1],padding='SAME')
def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * tf.abs(x)

class fullConnModel(object):
	def __init__(self,sess):
		self.session=sess
		self.fft_len=config.fft_len
		self.global_step=tf.Variable(0)
		self.best_loss_saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)
		self.saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)
		self.k_sparse=80
		self.h_range=800
		self.idx_offset_data=[]
		self.ext=20
		self.l_f=0
		self.f_n=8
		self.temp_fft_len=int(self.fft_len/self.f_n+self.ext*2)	
		for i in range(int(config.batch_size*config.frame_num)):
			temp1=[]
			for j in range(int(self.temp_fft_len)):
				temp2=tf.ones([1,self.k_sparse],tf.int32)*(i*self.temp_fft_len*self.h_range+j*self.h_range)
				temp1.append(temp2)
			self.idx_offset_data.append(temp1)
		self.create_placeholder()

		self.optimize()
	def reset(self):
		self.global_step=tf.Variable(0)
	def create_placeholder(self):
		self._input=tf.placeholder("float",shape=[None,None,64*self.fft_len])
		self._target=tf.placeholder("float",shape=[None,None,50*self.fft_len])
		self.keep_prob=tf.placeholder("float")
		# self.idx_offset_data=tf.placeholder("int32",shape=[None,None,1,self.k_sparse])
	def Conv_fully_net(self,inputs):
		self.batch_size=tf.shape(inputs)[0]
		self.frame_num=tf.shape(inputs)[1]
		# x_image=tf.reshape(inputs,[-1,self.fft_len*64])
		conv_out=[]
		x_image=tf.reshape(inputs,[-1,self.fft_len,1,64])
		w_convp1=weight_variable([1,1,64,100])
		b_convp1=bias_variable([100])
		# h_conv1 = conv2d(x_image, w_conv1) + b_conv1
		h_convp1=LeakyRelu(conv2d(x_image,w_convp1)+b_convp1)
		h_conv3_out=[]
		masked_conv3_out=[]
		temp_fft_len=self.temp_fft_len
		w3_add=[]
		w4_add=[]
		
		for f_list_ii in range(self.f_n):
			with tf.variable_scope("layer%d"%f_list_ii) as scope:
				if f_list_ii==0:
					t_h_convp1=h_convp1[:,0:temp_fft_len,:,:]
				else:
					if f_list_ii==self.f_n-1:
						t_h_convp1=h_convp1[:,(self.fft_len-temp_fft_len):self.fft_len,:,:]
					else:
						t_h_convp1=h_convp1[:,int(self.fft_len/self.f_n*f_list_ii-self.ext):int(self.fft_len/self.f_n*(f_list_ii+1)+self.ext),:,:]
				w_conv2=weight_variable([1,1,100,200])
				b_conv2=bias_variable([200])
				h_conv2=LeakyRelu(conv2d(t_h_convp1,w_conv2)+b_conv2)		

				w_conv21=weight_variable([1,1,200,400])
				b_conv21=bias_variable([400])
				h_t21 = conv2d(h_conv2,w_conv21)+b_conv21
				# h_t21=tf.contrib.layers.batch_norm(h_t21,center=True, scale=True, scope='bn21')
				h_conv21=LeakyRelu(h_t21)
				w_conv3=weight_variable([1,1,400,self.h_range])
				b_conv3=bias_variable([self.h_range])
				h_t3 = conv2d(h_conv21,w_conv3)+b_conv3
				# h_t3=tf.contrib.layers.batch_norm(h_t3,center=True, scale=True, scope='bn3')
				h_conv3=LeakyRelu(h_t3)
				h_conv3_out.append(tf.reshape(h_conv3,[self.batch_size,self.frame_num,self.h_range*temp_fft_len]))
				value_data,indices_data=tf.nn.top_k(h_conv3,self.k_sparse)
				# print(self.idx_offset_data)
				# print(indices_data.shape)
				indices_data+=self.idx_offset_data
				sparse_idx_data=tf.reshape(indices_data,[-1])
				sparse_val_data=tf.ones([self.batch_size*self.frame_num*temp_fft_len*self.k_sparse])
				mask_data=tf.sparse_to_dense(sparse_indices=sparse_idx_data,sparse_values=sparse_val_data,
											output_shape=[self.batch_size*self.frame_num*temp_fft_len*self.h_range],validate_indices=False)
				reshape_mask_data=tf.reshape(mask_data,[self.batch_size*self.frame_num,temp_fft_len,1,self.h_range])
				masked_conv3=tf.multiply(h_conv3,reshape_mask_data)
				# masked_conv3_out.append(masked_conv3)
				# masked_conv3_out=tf.concat(masked_conv3_out,2)
				w_conv4=weight_variable([1,1,self.h_range,400])
				b_conv4=bias_variable([400])
				h_t4=conv2d(masked_conv3,w_conv4)+b_conv4
				# h_t4=tf.contrib.layers.batch_norm(h_t4,center=True, scale=True, scope='bn4')
				h_conv4=LeakyRelu(h_t4)+h_conv21
				w_conv41=weight_variable([1,1,400,200])
				b_conv41=bias_variable([200])
				h_t41=conv2d(h_conv4,w_conv41)+b_conv41
				# h_t41=tf.contrib.layers.batch_norm(h_t41,center=True, scale=True, scope='bn41')
				h_conv41=tf.nn.tanh(h_t41)
				w_conv5=weight_variable([1,1,200,50])
				b_conv5=bias_variable([50])
				h_t5=conv2d(h_conv41,w_conv5)+b_conv5
				# h_t5=tf.contrib.layers.batch_norm(h_t5,center=True, scale=True, scope='bn5')
				h_conv5=tf.nn.tanh(h_t5)
				# if f_list_ii==0:
				# 	h_conv5_out=tf.reshape(h_conv5[:,0:temp_fft_len,:,:],[-1,50*(temp_fft_len-0)])
				# else:
				# 	if f_list_ii==self.f_n-1:
				# 		h_conv5_out=tf.reshape(h_conv5,[-1,50*(temp_fft_len)])	
				# 	else:
						# if f_list_ii==1 or f_list_ii==2:
				h_conv5_out=tf.reshape(h_conv5,[-1,50*(temp_fft_len)])
				conv_out.append(h_conv5_out)

				w3_add.append(w_conv3)
				w4_add.append(w_conv4)
		w3_add=tf.concat(w3_add,0)
		w4_add=tf.concat(w4_add,0)

		conv_out=tf.concat(conv_out,1)

		h_conv3_out=tf.concat(h_conv3_out,2)

		w_pool1=[]
		b_pool=[]
		fully_net_out1=[]
		for f_ii in range(temp_fft_len*self.f_n+self.l_f):
			with tf.variable_scope("fully_net%d"%f_ii) as scope:
				w_fc1=weight_variable([50,50])
				b_fc1=weight_variable([50])

				net_out1=tf.nn.tanh(tf.matmul(conv_out[:,50*f_ii:50*(f_ii+1)],w_fc1)+b_fc1)
				w_pool1.append(w_fc1)
				b_pool.append(b_fc1)
				fully_net_out1.append(net_out1)
		out=tf.concat(fully_net_out1,1)

		data_num=config.batch_size*config.frame_num
		weight11=tf.constant(np.arange(self.ext*50,0,-1.0,dtype='float32')/self.ext/50.0)
		weight11=tf.tile([weight11],(data_num,1))
		weight12=tf.constant(np.arange(0,self.ext*50,1.0,dtype='float32')/self.ext/50.0)
		weight12=tf.tile([weight12],(data_num,1))
		sig_out=[]
		for f_ii in range(self.f_n):
			if f_ii==0:
				sig_out.append(out[:,0:50*(temp_fft_len-self.ext*2)])
			else:
				if f_ii==self.f_n-1:
					temp1=tf.multiply(weight11,out[:,50*(temp_fft_len*f_ii-self.ext):50*(temp_fft_len*f_ii)])+\
						  tf.multiply(weight12,out[:,50*(temp_fft_len*f_ii+self.ext*2):50*(temp_fft_len*f_ii+self.ext*3)])
					sig_out.append(temp1)
					sig_out.append(out[:,50*(temp_fft_len*f_ii+self.ext*3):50*(temp_fft_len*self.f_n)])
				else:
					if f_ii==1:
						temp1=tf.multiply(weight11,out[:,50*(temp_fft_len*f_ii-self.ext*2):50*(temp_fft_len*f_ii-self.ext)])+\
						  tf.multiply(weight12,out[:,50*(temp_fft_len*f_ii+self.ext):50*(temp_fft_len*f_ii+self.ext*2)])
						sig_out.append(temp1)
						sig_out.append(out[:,50*(temp_fft_len*f_ii+self.ext*2):50*(temp_fft_len*(f_ii+1)-self.ext)])
					else:
						temp1=tf.multiply(weight11,out[:,50*(temp_fft_len*f_ii-self.ext):50*(temp_fft_len*f_ii)])+\
						  tf.multiply(weight12,out[:,50*(temp_fft_len*f_ii+self.ext):50*(temp_fft_len*f_ii+self.ext*2)])
						sig_out.append(temp1)
						sig_out.append(out[:,50*(temp_fft_len*f_ii+self.ext*2):50*(temp_fft_len*(f_ii+1)-self.ext)])
		sig_out=tf.concat(sig_out,1)


		pred=[]
		pred.append(tf.reshape(out,[self.batch_size,self.frame_num,50*(self.temp_fft_len*self.f_n)]))
		pred.append(h_conv3_out)
		pred.append(tf.reshape(sig_out,[self.batch_size,self.frame_num,50*self.fft_len]))
		return pred

	def optimize(self):
		self.lr0=0.003
		self.lr_decay=0.9994
		self.lr_step=15
		self.lr=tf.train.exponential_decay(
			self.lr0,
			self.global_step,
			decay_steps=self.lr_step,
			decay_rate=self.lr_decay,
			staircase=True)
		optimizer=tf.train.AdamOptimizer(self.lr)
		# pred=self.Three_layer_fully_conn(self._input)
		pred=self.Conv_fully_net(self._input)

		prs_tar=[]
		for f_ii in range(self.f_n):
			if f_ii==0:
				prs_tar.append(self._target[:,:,50*0:50*self.temp_fft_len])
			else:
				if f_ii==self.f_n-1:
					prs_tar.append(self._target[:,:,50*(self.fft_len-self.temp_fft_len):50*self.fft_len])
				else:
					prs_tar.append(self._target[:,:,50*int(self.fft_len/self.f_n*f_ii-self.ext):50*int(self.fft_len/self.f_n*(f_ii+1)+self.ext)])
		prs_tar=tf.concat(prs_tar,2)


		error=pred[0]-prs_tar
		mse_loss1=tf.reduce_mean(error**2)
		mse_loss2=tf.reduce_mean((pred[1])**2)
		loss=mse_loss2*8+mse_loss1*1
		self.train_step=optimizer.minimize(loss,global_step=self.global_step)
		self.pred=pred[2]
		self.loss=loss
		self.loss1=mse_loss1
	def run_batch(self,train_data):

		# hpool1,hpool2,hdrop=self.session.run([self.h_pool1,self.h_pool2,self.h_fc1_drop],
		# 	feed_dict={self._input:train_data[0],self._target:train_data[1],self.keep_prob:0.8})
		


		#original
		_,i_loss,i_lr,i_global_step=self.session.run([self.train_step,self.loss,self.lr,self.global_step],
			feed_dict={self._input:train_data[0],self._target:train_data[1],self.keep_prob:0.9})
		# print(i_mask)
		# print(i_mask.shape)
		return i_loss, i_lr,i_global_step
	def get_pred(self,test_data):
		pred=self.session.run(self.pred,feed_dict={self._input:test_data,self.keep_prob:1})
		return pred
	def check_loss(self,dev_data):
		loss=self.session.run(self.loss1,feed_dict={self._input:dev_data[0],self._target:dev_data[1],
			                                         self.keep_prob:1})
		return loss
	def valid(self,reader):
		total_loss,batch_counter=0.0,0
		logging.info("start to dev")
		while True:
			batch_data=reader.next_batch()
			# print(reader.batch_size)
			if batch_data==None:
				reader.reset()
				break
			else:
				loss=self.session.run(self.loss1,feed_dict={self._input:batch_data[0],self._target:
					                 batch_data[1],self.keep_prob:1})
				total_loss+=loss
				batch_counter+=1
				if batch_counter%15==0:
					print('dev_batch:',batch_counter,"loss: ",total_loss/batch_counter)
		if batch_counter>0:		
			avg_loss=total_loss/batch_counter
		else:
			avg_loss=0
		return avg_loss






		
