import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.layers as tcl
import time
from speech_data import data_reader
import config
import matplotlib.pyplot as plt
import os
from neural_net import fullConnModel
import logging
from tools import MetricChecker
from write_list import get_file_list
from scipy.io import loadmat
from scipy.io import savemat
from tensorboardX import SummaryWriter
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
#import tkinter

if __name__=="__main__":
	# out folde
	root_path = 'gaoData'
	# model save path
	folder= os.path.join(root_path, 'model', 'gao_net')
	if not os.path.exists(folder):
		os.makedirs(folder)	
	# mic_file:input signal folder of the network
	# hoa_file:output signal folder of the network
	train_path = os.path.join(root_path, 'signal', 'train', 'uniform_mic_noisy30dB')
	mic_file= os.path.join(train_path, 'mic_sig')
	hoa_file= os.path.join(train_path, 'hoa_sig')
	single_snr=True
	file_list_reader=get_file_list(single_snr)
	train_mic_list,train_hoa_list,dev_mic_list,dev_hoa_list,test_mic_list=file_list_reader.get_train_list(mic_file,hoa_file)
	train_reader=data_reader(train_mic_list,train_hoa_list,job_type='train')
	dev_reader=data_reader(dev_mic_list,dev_hoa_list,job_type='dev')
	test_reader=data_reader(test_mic_list,test_mic_list,job_type='test')

	writer = SummaryWriter(os.path.join(root_path, 'run', 'gao_net'))
	# os.environ['CUDA_VISIBLE_DEVICES']='1'

	# per_process_gpu_memory_fraction= 1
	gpu_options = tf.GPUOptions(allow_growth=True)
	gpuConfig=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
	loss_checker=MetricChecker(config)
	# gpuConfig.gpu_options.per_process_gpu_memory_fractio=0.5
	sess=tf.InteractiveSession(config=gpuConfig)

	model=fullConnModel(sess)
	sess.run(tf.global_variables_initializer())
	loss_list=[]


	# fig=plt.figure()
	# ax=fig.add_subplot(1,1,1)
	# plt.ion()
	batch_num=0
	best_num=0
	epoch_loss = []
	loss_filepath = os.path.join(root_path, 'loss', 'Net')
	np.save(loss_filepath, np.array(epoch_loss))
	for i_epoch in range(200):
		
		# break
		try:
			while not loss_checker.should_stop():
				train_data=train_reader.next_batch()
				# print(len(train_data))
				if train_data==None:
					train_reader.reset()
					# model.reset()
					break
				else:
					batch_num+=1
					start_time=time.time()
					frame_num=train_data[0].shape[0]
					#liurui
					i_loss,i_lr,i_global_step=model.run_batch(train_data)
					# print(model.get_pred(train_data).shape)
					# break
					loss_list.append(i_loss)
					end_time=time.time()
					if batch_num % 60 ==0:   #print the loss every 10 speech
						print('epoch: ',i_epoch,"; loss: ",i_loss,"; batch_num: ", batch_num,
							"; learing rate: ",i_lr)
					writer.add_scalar('loss', i_loss, batch_num)

					if batch_num % 1800 ==0:
						avg_loss=model.valid(dev_reader)
						loss_imporved,best_loss=loss_checker.update(sess,avg_loss)
						print(loss_imporved)
						if loss_imporved:
							# logging.info("new best loss {}".format(best_loss))
							print("new best loss:",best_loss)
							if best_num % 3 ==0:
								model_path=os.path.join(folder,'model.ckpt')
								save=tf.train.Saver(tf.global_variables())
								save.save(sess,model_path)
							best_num=best_num+1
							writer.add_scalar('valid_loss', avg_loss, best_num)
			avg_loss=model.valid(dev_reader)
			epoch_loss.append(avg_loss)
			
			np.save(loss_filepath, np.array(epoch_loss))

		except Exception as e:
			print(e)
			dev_reader._producer.stop()
			train_reader._producer.stop()	
		if loss_checker.should_stop():
			print("early stoped")
			# break

	dev_reader._producer.stop()
	train_reader._producer.stop()
	# plt.show()
	print('finish')	
	model_path=os.path.join(folder,'model.ckpt')
	save=tf.train.Saver(tf.global_variables())
	save.save(sess,model_path)
	print("finish")

