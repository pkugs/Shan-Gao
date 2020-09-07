## 1 Environment configuration
1. computer with GPU
2. matlab
3. python	
	* numpy	
	* scipy	
	* tensorboardX	
	* tensorflow	
	* matplotlib	
		
## 2 Run steps

step1: generate dataset	 (enter data_generate folder)
> (1) generate microphone channels transfer function	
		`
		run data_generate/creat_tf/creatMicTF.m # Modify the data storage path in creatMicTF.m 
		` 	
(2) generate HOA channels transfer function			`
		run data_generate/creat_tf/creatHoaTF.m # Modify the data storage path in creatHoaTF.m
		` 	
(3) generate dataset for train
		`
		run data_generate/creat_signal/creatTrainSignal.m # Modify the data storage path in creatTrainSignal.m
		` 
		
step2: train network (enter network folder)
		`
		run network/main.py # Modify the train data path in main.py
		` 