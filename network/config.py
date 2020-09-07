import numpy as np
from scipy.io import loadmat
from scipy import signal
from scipy.io import wavfile
import config

frame_len=1024
fft_len=512
frame_num=20
sample_rate=48000
early_stop_count=200
train_ratio=0.9
dev_ratio=0.05
test_ratio=0.05
batch_size= 4
min_queue_size=6

band_number=8
