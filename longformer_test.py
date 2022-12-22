# Import tensorflow and transformer models fro huggingface
import tensorflow as tf

# Get and print GPU info to ensure GPU usage
gpus = tf.config.list_physical_devices('GPU')
gpu_info = tf.config.experimental.get_device_details(gpus[0])
print('Device name: ', gpu_info['device_name'])

