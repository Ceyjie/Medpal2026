import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL_PATH = 'Mobilefacenet-TF2-coral_tpu/pretrained_model/edgetpu_v1/mobilefacenet_edgetpu.tflite'
REF_EMBEDDING_PATH = 'reference_embedding.npy'
IMAGE_PATH = 'reference.jpg'

# Initialize the Edge TPU interpreter
print("Initializing Edge TPU interpreter...")
delegate = load_delegate('libedgetpu.so.1')
interpreter = Interpreter(MODEL_PATH, experimental_delegates=[delegate])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_h, input_w = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]['dtype']

# Check for a static reference image, otherwise capture from webcam
if os.path.exists(IMAGE_PATH):
    print(f"Loading reference image from {IMAGE_PATH}...")
    frame = cv2.imread(IMAGE_PATH)
else:
    print("No 'reference.jpg' found. Opening camera to capture reference frame...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open video stream from camera.")

    print("Position your target in front of the camera. Press 's' to capture and save, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        cv2.imshow("Capture Reference - Press 's' to Save", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(IMAGE_PATH, frame)
            print("Reference image saved as reference.jpg")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    cap.release()
    cv2.destroyAllWindows()

# Preprocess the reference frame
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb_frame, (input_w, input_h))

if input_dtype == np.uint8:
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
else:
    input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

# Run inference to generate embedding vector
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

embedding = interpreter.get_tensor(output_details[0]['index']).flatten().astype(np.float32)

# Normalize the embedding vector
norm = np.linalg.norm(embedding)
if norm > 0:
    embedding = embedding / norm

# Save to disk
np.save(REF_EMBEDDING_PATH, embedding)
print(f"Success! Reference embedding saved to {REF_EMBEDDING_PATH}")