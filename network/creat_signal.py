import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from scipy import signal
from scipy.io import wavfile
import config
import math
import matplotlib.pyplot as plt
import time
class signal_process(object):
	def __init__(self, config):
		self.frame_len=config.frame_len
		self.shift_len=config.shift_len
	def read_file(self,wav_name):
		sample_rate,wav_signal=wavfile.read(wav_name)
		self.sample_rate=sample_rate
		wav_signal=wav_signal/(2.**15)
		return wav_signal
	def stft_trans(self,wav_name):
		wav_signal=self.read_file(wav_name)
		# frame_num=int(math.floor((len(wav_signal)-self.frame_len)/self.shift_len));
		# print(frame_num)
		# wav_signal=wav_signal[0:frame_num*self.shift_len]
		out_fft_signal=[]
		for ch_i in range(wav_signal.shape[1]):
			f,t,fft_signal=signal.stft(wav_signal[:,ch_i],fs=self.sample_rate,
				                       nperseg=self.frame_len)
			out_fft_signal.append(fft_signal)
		out=np.array(out_fft_signal)
		return out
	def stft_to_input(self,signal):
		[ch_num,freq_num,frame_num]=signal.shape
		signal_out=[]
		for frame_ii in range(frame_num):
			temp_data_freq=signal[:,1:freq_num,frame_ii]
			comp_data=np.transpose(temp_data_freq).ravel()
			real_data=np.transpose([np.real(comp_data),np.imag(comp_data)]).ravel()
		signal_out.append(real_data)

			

		# for frame_ii in range(frame_num):
		# 	temp_data=signal[frame_ii*shift_len:frame_ii*shift_len+frame_len]

if __name__=="__main__":
	wav_name="/media/G/gs/mic_to_hoa_with_nn/signals/train_signals/hoa_signal/01aa010f.wav"
	trans_func=signal_process(config)
	inii_signal=trans_func.read_file(wav_name)
	print(inii_signal.shape)
	out=trans_func.stft_trans(wav_name)
	print(len(out))
	print(out.shape)
	# _,ini_signal=signal.istft(out)
	# print(ini_signal.shape)
	# fs=16000
	# plt.figure(1)
	# N=len(inii_signal)
	# time_list=np.arange(N)/float(fs)
	# plt.plot(time_list,inii_signal)

	# plt.figure(2)
	# N=len(ini_signal)
	# time_list=np.arange(N)/float(fs)
	# plt.plot(time_list,ini_signal)
	# plt.show()
	# plt.close()
	# print(out[:,1])



