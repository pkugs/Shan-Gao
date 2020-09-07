import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft,ifft
try: 
	from Queue import Queue
except ImportError:
	from queue import queue
import random
import matplotlib.pyplot as plt
import config
class data_reader(object):
	def __init__(self,config,tf_file,hoa_file):
		self._config=config
		self.tf_file=tf_file
		self.hoa_file=hoa_file
		self.sample_buffer_train=[]
		self.sample_buffer_dev=[]
		self.sample_buffer_test=[]
		self.load_data()
		self.reset()
		
	def reset(self):
		
		self.next_load_idx=0
		random.shuffle(self.sample_buffer_train)

	def load_data(self):
		data=loadmat(self.tf_file)
		tf=data['data']
		tf=np.squeeze(tf)
		s_num=len(tf)
		self.s_num=s_num
		[sig_len,tf_ch_num]=tf[0].shape
		prs_tf=np.zeros((s_num,self._config.fft_len,tf_ch_num))*complex(1,1)
		for s_ii in range(s_num):
			for ch_ii in range(tf_ch_num):
				ch_data=tf[s_ii][:,ch_ii]
				fft_ch_data=fft(ch_data)
				prs_tf[s_ii,:,ch_ii]=fft_ch_data[1:self._config.fft_len+1]

		data=loadmat(self.hoa_file)
		hoa=data['data']
		hoa=np.squeeze(hoa)
		[sig_len,hoa_ch_num]=hoa[0].shape
		prs_hoa=np.zeros((s_num,self._config.fft_len,hoa_ch_num))*complex(1,1)
		for s_ii in range(s_num):
			for ch_ii in range(hoa_ch_num):
				ch_data=hoa[s_ii][:,ch_ii]
				fft_ch_data=fft(ch_data)
				prs_hoa[s_ii,:,ch_ii]=fft_ch_data[1:self._config.fft_len+1]


		sample_buffer=[]
		for s_ii in range(s_num):
			temp1=prs_tf[s_ii,:,:]
			temp2=temp1.ravel()
			sin_prs_tf=np.transpose([np.real(temp2),np.imag(temp2)]).ravel()

			temp1=prs_hoa[s_ii,:]
			temp2=temp1.ravel()
			sin_prs_hoa=np.transpose([np.real(temp2),np.imag(temp2)]).ravel()

			sample_buffer.append([sin_prs_tf,sin_prs_hoa])

		self.sample_buffer_train=sample_buffer[0:int(s_num*self._config.train_ratio)]
		self.sample_buffer_dev=sample_buffer[int(s_num*self._config.train_ratio):int(s_num*self._config.train_ratio)+int(s_num*self._config.dev_ratio)]
		self.sample_buffer_test=sample_buffer[int(s_num*self._config.train_ratio)+int(s_num*self._config.dev_ratio):s_num]

		# return tf
	def patch_data(self,batch_num):
		batch_mic=np.zeros((batch_num,self._config.fft_len*64))
		batch_hoa=np.zeros((batch_num,self._config.fft_len*50))
		for ii in range(batch_num):
			batch_mic[ii,:]=self.sample_buffer_train[self.next_load_idx+ii][0]
			batch_hoa[ii,:]=self.sample_buffer_train[self.next_load_idx+ii][1]
		self.next_load_idx+=batch_num
		batch_data=[batch_mic,batch_hoa]
		return batch_data
	def next_batch_train(self,batch_num):
		if self.next_load_idx+batch_num<=len(self.sample_buffer_train):
			out_data=self.patch_data(batch_num)
			return out_data
		else:
			return None

	def next_batch_dev(self):
		batch_num=len(self.sample_buffer_dev)
		batch_mic=np.zeros((batch_num,self._config.fft_len*64))
		batch_hoa=np.zeros((batch_num,self._config.fft_len*50))
		for ii in range(batch_num):
			batch_mic[ii,:]=self.sample_buffer_dev[ii][0]
			batch_hoa[ii,:]=self.sample_buffer_dev[ii][1]
		batch_data=[batch_mic,batch_hoa]
		return batch_data

	def next_batch_test(self):
		batch_num=len(self.sample_buffer_test)
		batch_mic=np.zeros((batch_num,self._config.fft_len*64))
		for ii in range(batch_num):
			batch_mic[ii,:]=self.sample_buffer_test[ii][0]
		return batch_mic

	def test_batch_to_mat(self,batch_data,ch_num):
		batch_data=np.squeeze(batch_data)
		[batch_num,all_ch_sig_len]=batch_data.shape
		out=[]
		for batch_ii in range(batch_num):
			temp1=batch_data[batch_ii,:]
			temp1=temp1.reshape(all_ch_sig_len/2,2)
			temp2=np.zeros((all_ch_sig_len/2))*complex(1,1)
			for f_ii in range(all_ch_sig_len/2):
				temp2[f_ii]=complex(temp1[f_ii,0],temp1[f_ii,1])
			temp2=temp2.reshape(self._config.fft_len,ch_num)
			temp3=np.zeros((self._config.frame_len,ch_num))
			for ch_ii in range(ch_num):
				temp4=np.zeros((self._config.frame_len))*complex(1,1)
				temp4[1:self._config.fft_len]=temp2[0:self._config.fft_len-1,ch_ii]
				temp4[self._config.fft_len+1:self._config.frame_len]=np.flipud(temp2[0:self._config.fft_len-1,ch_ii]).conjugate()
				temp3[:,ch_ii]=ifft(temp4)
			out.append(temp3)
		return out




if __name__=="__main__":
	tf_file='/data/gs/mic_to_hoa_new/signal/train/TF-1024-r042cm-48khz-with-2000p.mat'
	hoa_file='/data/gs/mic_to_hoa_new/signal/train/HOA-1024-r042cm-48khz-with-2000p.mat'
	reader=data_reader(config,tf_file,hoa_file)
	tf=reader.load_data()
	data=reader.next_batch_test()
	out=reader.test_batch_to_mat(data,32)
	out1=out[0][:,1]
	tf1=tf[1800][:,2]
	plt.plot(out1-tf1)
	# plt.plot(out[0][:,1])
	plt.show()
