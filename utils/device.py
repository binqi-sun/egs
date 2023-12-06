import tensorflow as tf

def config_device(gpu_id):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpu = gpus[gpu_id]
            tf.config.set_visible_devices(gpu, 'GPU')
            print(f"EGS: GPU {gpu_id} is going to be used.")
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"EGS: {e}")
    