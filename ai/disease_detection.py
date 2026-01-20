import cv2
import numpy as np
from utils.tflite_loader import Interpreter

MODEL_PATH = "models/disease_model.tflite"


def detect_disease(image_file):
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])
    classes = ["Healthy", "Disease"]
    return classes[int(np.argmax(prediction))]
