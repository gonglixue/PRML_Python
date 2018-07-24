import tensorflow as tf
import os
import sys
from datetime import datetime
import numpy as np

# device_name = sys.argv[1]
# shape = (int(sys.argv[2]), int(sys.argv[2]))

shape = (5000, 5000)
device_name = "/gpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
    var_list = [var for var in tf.global_variables()]
    result = session.run(sum_operation)
    print(result)

print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time cost: ", datetime.now() - startTime)

