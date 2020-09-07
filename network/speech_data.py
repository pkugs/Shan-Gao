import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from scipy.fftpack import fft,ifft
from scipy import signal
from scipy.io import wavfile
import config
import math
import matplotlib.pyplot as plt
import time
import os, sys
import random
import logging
import threading
import traceback
try:
	from Queue import Queue
except ImportError:
	from queue import Queue
	# import queue
# import librosa


class Producer(threading.Thread):
	def __init__(self,reader):
		threading.Thread.__init__(self)
		self.reader=reader
		self.exitcode=0
		self.stop_flag=False
	def run(self):
		try:
			min_queue_size=self.reader._config.min_queue_size
			while not self.stop_flag:
				idx=self.reader.next_load_idx
				if idx+self.reader.batch_size>=len(self.reader.data_list):
					self.reader._batch_queue.put([])
					break
				if self.reader._batch_queue.qsize()<min_queue_size:
					batch=self.reader.load_samples2()
					self.reader._batch_queue.put(batch)
				else:

					time.sleep(1)
		except Exception as e:
			self.exitcode=1
			traceback.print_exc()
	def stop(self):
		self.stop_flag=True

class data_reader(object):
	def __init__(self,mic_file_list,hoa_file_list,job_type):
		self.frame_len=config.frame_len
		self.batch_size=config.batch_size
		self._config=config
		self.job_type=job_type
		self.wav_num=0
		self.data_list=list(zip(mic_file_list,hoa_file_list))
		self.sample_rate=config.sample_rate
		if (job_type=='train')| (job_type=='dev'):
			# print(job_type)
			# print('nononon')
			self._batch_queue=Queue()
			self.reset()

	# def get_data_list(self,data_list_path):
	# 	mic_file_path,hoa_file_path=data_list_path
	# 	mic_file_list=self.read_file_line(mic_file_path)
	# 	hoa_file_list=self.read_file_line(hoa_file_path)
	# 	tuple_list=list(zip(mic_file_list,hoa_file_list))
		# return tuple_list
	def reset(self):
		self.sample_buffer=[]
		self.next_load_idx=0
		if self.job_type=="train":
			self.shuffle_data_list()
		self._producer=Producer(self)
		self._producer.start()
	def shuffle_data_list(self):
		random.shuffle(self.data_list)
	def read_all_ch_sig(self,file_name):
		signal=self.read_file(file_name)
		fft_signal=self.stft_trans(signal)
		out=self.stft_to_input(fft_signal)
		return out
	def read_file(self,wav_name):
		sample_rate,wav_signal=wavfile.read(wav_name)
		self.sample_rate=sample_rate
		# wav_signal=wav_signal/(2.**15)
		# wav_signal=wav_signal/(2.**8)
		return wav_signal
	def stft_trans(self,wav_signal):
		out_fft_signal=[]
		for ch_i in range(wav_signal.shape[1]):
			f,t,fft_signal=signal.stft(wav_signal[:,ch_i],fs=self.sample_rate,
										nperseg=self.frame_len)
			out_fft_signal.append(fft_signal)
		out=np.array(out_fft_signal)
		return out
	def stft_to_input(self,signal):
		[ch_num,freq_num,frame_num]=signal.shape
		signal_out=np.zeros((frame_num,ch_num*2*(freq_num-1)))
		for frame_ii in range(frame_num):
			temp_data_freq=signal[:,0:freq_num-1,frame_ii]
			comp_data=np.transpose(temp_data_freq).ravel()
			signal_out[frame_ii,:]=np.transpose([np.real(comp_data),np.imag(comp_data)]).ravel()
		return signal_out 
	def read_mat_sig(self,file):
		data=loadmat(file)
		out=data['data']
		return out
	def load_samples2(self):
		batch_size=self.batch_size
		idx=self.next_load_idx
		batch_mic=np.zeros((batch_size,config.frame_num,int(32*config.frame_len)))
		batch_hoa=np.zeros((batch_size,config.frame_num,int(25*config.frame_len)))
		for ii in range(batch_size):
			tuple_file=self.data_list[idx+ii]
			t_mic=self.read_mat_sig(tuple_file[0])
			t_hoa=self.read_mat_sig(tuple_file[1])
			for jj in range(len(t_mic[:,1])):
				t_max=np.max(np.abs(t_mic[jj,:]))
				t_mic[jj,:]=t_mic[jj,:]/t_max/2
				t_hoa[jj,:]=t_hoa[jj,:]/t_max/2
			batch_mic[ii,:,:]=t_mic
			batch_hoa[ii,:,:]=t_hoa
		self.next_load_idx+=batch_size
		batch_data=[batch_mic,batch_hoa]
		return batch_data


	def next_batch(self):
		while self._producer.exitcode==0:
			try:
				batch_data=self._batch_queue.get(block=False)
				# batch_data=self.load_samples2()
				if len(batch_data)==0:
					return None
				else:
					return batch_data
			except Exception as e:
				time.sleep(3)
	def output_to_signal(self,output,channel_num):
		output=np.squeeze(output)
		[frame_num,out_len]=output.shape
		# out=np.zeros((frame_num*self.frame_len,channel_num))
		for channel_ii in range(channel_num):
			fft_channel_signal=np.zeros((self.frame_len/2+1,frame_num))
			fft_channel_signal=fft_channel_signal*complex(1,1)
			for frame_ii in range(frame_num):
				for sig_ii in range(1,self.frame_len/2):
					fft_channel_signal[sig_ii,frame_ii]=complex(output[frame_ii,sig_ii*channel_num*2+channel_ii*2],
					                    output[frame_ii,sig_ii*channel_num*2+channel_ii*2+1])
			_,channel_signal=signal.istft(fft_channel_signal,fs=self.sample_rate,nperseg=self.frame_len)
			if channel_ii==0:
				out=np.zeros((len(channel_signal),channel_num))
				out[:,0]=channel_signal
			else:
				out[:,channel_ii]=channel_signal
		return out


class mat_data_reader(object):
	def __init__(self,mic_file,hoa_file):
		self.s_signal=self.creat_signal()
		self.mic_data=self.load_data(mic_file)
		self.hoa_data=self.load_data(hoa_file)
		# self.package_data()
		self.reset()
	def reset(self):
		self.next_load_idx=0



	def load_data(self,file_name):
		data=loadmat(file_name)
		signal=data['data']
		return signal
	# def record_signal(self,sta_pot,end_pot,tf_type):
	# 	if tf_type=='mic_tf':
	# 		signal=self.mic_data[sta_pot:end_pot]
	# 	else:
	# 		signal=self.hoa_data[sta_pot:end_pot]

	# 	frame_num=len(signal)
	# 	[frame_len,ch_num]=signal[0].shape
	# 	out=np.zeros((frame_num,100,config.fft_len*2*ch_num))
	# 	for frame_ii in range(frame_num):
	# 		for sig_frame_ii in range(100):
	# 			frame_out=np.zeros((config.fft_len,ch_num))*complex(1,1)
	# 			for ch_ii in range(ch_num):
	# 				temp1=fft(signal[frame_ii][:,ch_ii])
	# 				temp2=temp1*self.s_signal[:,sig_frame_ii,frame_ii+sta_pot]
	# 				frame_out[:,ch_ii]=temp2[0:config.fft_len]
	# 			temp3=frame_out.ravel()
	# 			out[frame_ii,sig_frame_ii,:]=np.transpose([np.real(temp3),np.imag(temp3)]).ravel()
	# 	return out
	# def package_data(self):
	# 	for frame_ii in range()

	# def patch_data(self,sta_pot,end_pot):
	# 	mic_signal=self.record_signal(sta_pot,end_pot,tf_type='mic_tf')
	# 	hoa_signal=self.record_signal(sta_pot,end_pot,tf_type='hoa_tf')
	# 	out=[mic_signal,hoa_signal]
	def patch_data(self,sta_pot,end_pot):
		mic_signal=self.mic_data[sta_pot:end_pot,:,:]
		hoa_signal=self.hoa_data[sta_pot:end_pot,:,:]
		out=[mic_signal,hoa_signla]
		return out

	def next_batch_train(self,batch_num):

		if self.next_load_idx+batch_num<=len(self.mic_data[:,0,0])*config.train_ratio:
			out=self.patch_data(self.next_load_idx,self.next_load_idx+batch_num)
			# out=[self.mic_data[self.next_load_idx:self.next_load_idx+batch_num,:],
			#      self.hoa_data[self.next_load_idx:self.next_load_idx+batch_num,:]]
			self.next_load_idx+=batch_num
			return out
		else:
			return None
	def next_batch_dev(self):
		frame_len=len(self.mic_data[:,0,0])
		sta_pot=int(frame_len*config.train_ratio)
		end_pot=int(frame_len*(config.train_ratio+config.dev_ratio))
		out=self.patch_data(sta_pot,end_pot)
		# out=[self.mic_data[sta_pot:end_pot,:],
		# 	 self.hoa_data[sta_pot:end_pot,:]]
		return out
	def next_batch_test(self):
		frame_len=len(self.mic_data[:,0,0])
		sta_pot=int(frame_len*(config.train_ratio+config.dev_ratio))
		temp_out=self.patch_data(sta_pot,frame_len)
		out=temp_out[0]
		# out=self.mic_data[sta_pot:frame_len,:]
		return out
	def out_to_mat(self,data,ch_num):
		[frame_num,all_frame_len]=data.shape
		out=np.zeros((frame_num,config.frame_len,ch_num))
		for frame_ii in range(frame_num):
			for ch_ii in range(ch_num):
				temp_ch=np.zeros((config.frame_len))*complex(1,1)
				for sig_ii in range(1,config.fft_len):
					temp_ch[sig_ii]=complex(data[frame_ii,sig_ii*ch_num*2+ch_ii*2],
						data[frame_ii,sig_ii*ch_num*2+ch_ii*2+1])
					temp_ch[config.frame_len-sig_ii]=complex(data[frame_ii,sig_ii*ch_num*2+ch_ii*2],
						-data[frame_ii,sig_ii*ch_num*2+ch_ii*2+1])
				out[frame_ii,:,ch_ii]=ifft(temp_ch)
		return out


if __name__=="__main__":
	file='/data/gs/mic_to_hoa_new/signal/train/hoa_sig/1.mat'
	data=loadmat(file)
	sig=data['data']
	print(sig.shape)


