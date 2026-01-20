try:
    import tflite_runtime.interpreter as tflite

    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
