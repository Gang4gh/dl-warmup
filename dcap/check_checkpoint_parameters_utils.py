import sys
import tensorflow as tf
import numpy as np
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python.training import py_checkpoint_reader

f0 = 'iter9-parity-fullattn-padding-mask-adjust-1217-noTH2s-r0.1-20200311-101704'
f1 = 'iter9-parity-fullattn-padding-mask-adjust-dropout-1217-noTH2s-r0.1-20200311-201403'
f2 = 'iter9-parity-lshattn-padding-mask-hash2-1217-noTH2s-r0.1-20200311-101815'

print(sys.argv)

latest_ckp = tf.train.latest_checkpoint(f'running_center/{f2}/model')
np.set_printoptions(threshold=1024*10)
#print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
reader = py_checkpoint_reader.NewCheckpointReader(latest_ckp)
var_to_shape_map = reader.get_variable_to_shape_map()
var_to_dtype_map = reader.get_variable_to_dtype_map()
for key, value in sorted(var_to_shape_map.items()):
	print("tensor: %s (%s) %s" % (key, var_to_dtype_map[key].name, value))
	print(reader.get_tensor(key))
