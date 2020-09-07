import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.layers as tcl
import time
from speech_data import data_reader
import config
import matplotlib.pyplot as plt
import os
from neural_net import fullConnModel
from scipy.io import wavfile
from write_list import search_file
from scipy.io import loadmat
from scipy.io import savemat

if __name__=="__main__":
	root_path = 'gaoData'
	folder= os.path.join(root_path, 'model', 'gao_net80')
	gpuConfig=tf.ConfigProto(allow_soft_placement=True)
	sess=tf.InteractiveSession(config=gpuConfig)
	model_file=tf.train.latest_checkpoint(folder)
	print(model_file)
	model=fullConnModel(sess)
	saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)
	print(model_file)
	saver.restore(sess,model_file)
	test_root_folder = os.path.join(root_path, 'signal', 'test', 'clean')
	pred_root_folder = os.path.join(root_path, 'signal', 'pred', 'clean')
	if not os.path.exists(pred_root_folder):
		os.makedirs(pred_root_folder)	
	source_num_list=['single_source','double_source','triple_source','fourth_source']
	for source_num in source_num_list:
		test_folder= os.path.join(test_root_folder, source_num, 'mic_sig')
		file_list=search_file(test_folder)
		test_reader=data_reader(file_list,file_list,job_type='test')

		out_folder= os.path.join(pred_root_folder, source_num)
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)	
		for file_ii in file_list:
			# print(file_ii)
			name=os.path.basename(file_ii)
			_input=test_reader.read_mat_sig(file_ii)

			t_var=np.zeros([len(_input[:,0])])
			t_mean=np.zeros([len(_input[:,0])])
			t_max=np.zeros([len(_input[:,0])])
			for ii in range(len(_input[:,0])):
				t_max[ii]=np.max(np.abs(_input[ii,:]))*2
				_input[ii,:]=_input[ii,:]/t_max[ii]

			_input2=np.array([_input])
			pred=model.get_pred(_input2)
			pred=np.squeeze(pred)
			for ii in range(len(_input[:,0])):
				pred[ii,:]=pred[ii,:]*t_max[ii]

			out_name=os.path.join(out_folder,name)
			print(file_ii)
			savemat(out_name,{'data':pred})

# plt.show()

