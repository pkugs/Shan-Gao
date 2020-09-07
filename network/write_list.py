import os
import numpy as np 
import math
import random
import config
def search_file(file_dir):
	L=[];
	for root,dirs,files in os.walk(file_dir):
		for file in files:
			if os.path.splitext(file)[1]=='.mat'or os.path.splitext(file)[1]=='.wav':
				L.append(os.path.join(root,file))
	return L
def read_file(file_name):
	L=[];
	with open(file_name,'r') as f:
		for line in f.readlines():
			line=line.strip('\n')
			L.append(line)
	f.close()
	return L
def write_file(file_name,file_data):
	with open(file_name,'w') as f:
		for data in file_data:
			temp_data=data+'\n'
			f.write(temp_data)
	f.close()
class get_file_list(object):
	def __init__(self,single_snr):
		self.single_snr=single_snr

	def get_train_list(self,mic_dir,hoa_dir):
		mic_files=search_file(mic_dir)
		file_len=len(mic_files)
		# print(file_len)
		random.shuffle(mic_files)
		hoa_files=[]
		for data in mic_files:
			wav_name=os.path.basename(data)
			snr_name=os.path.basename(os.path.dirname(data))
			if self.single_snr:
				hoa_files.append(os.path.join(hoa_dir,wav_name))
			else:
				hoa_files.append(os.path.join(hoa_dir,snr_name,wav_name))
		dev_sta=int(math.floor(file_len*config.train_ratio))
		tst_sta=int(math.floor(file_len*(config.train_ratio+config.dev_ratio)))
		# print(dev_sta)
		# print(tst_sta)
		train_mic_list=mic_files[0:dev_sta]
		train_hoa_list=hoa_files[0:dev_sta]
		dev_mic_list=mic_files[dev_sta:tst_sta]
		dev_hoa_list=hoa_files[dev_sta:tst_sta]
		test_mic_list=mic_files[tst_sta:file_len]
		return train_mic_list,train_hoa_list,dev_mic_list,dev_hoa_list,test_mic_list
	def get_test_list(self,mic_dir):
		file_list=search_file(mic_dir)
		return file_list

if __name__=='__main__':
	mic_file=''
	hoa_file=''
	single_snr=True
	file_list_reader=get_file_list(single_snr)
	train_mic_list,train_hoa_list,dev_mic_list,dev_hoa_list,test_mic_list=file_list_reader.get_train_list(mic_file,hoa_file)
	print(len(train_mic_list))



