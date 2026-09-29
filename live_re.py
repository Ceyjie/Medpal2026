import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration constants
REF_EMBEDDING_PATH = 'reference_embedding.npy'
MODEL_PATH = 'Mobilefacenet-TF2-coral_tpu/pretrained_model/edgetpu_v1/mobilefacenet_edgetpu.tflite'
SIMILARITY_THRESHOLD = 0.70  # Adjust based on your embedding model's strictness

# 1. Validate and load reference embedding
if not os.path.exists(REF_EMBEDDING_PATH):
    raise FileNotFoundError("Reference embedding file not found. Run 'enroll_face.py' first.")
ref_embedding = np.load(REF_EMBEDDING_PATH)

# 2. Initialize the Edge TPU interpreter via tflite_runtime
print("Initializing Edge TPU interpreter...")
try:
    delegate = load_delegate('libedgetpu.so.1')
    interpreter = Interpreter(MODEL_PATH, experimental_delegates=[delegate])
    print("Edge TPU delegate loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load Edge TPU delegate ({e}), falling back to CPU interpreter.")
    interpreter = Interpreter(MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract model input dimensions dynamically
input_shape = input_details[0]['shape']
input_h, input_w = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]['dtype']

# 3. Start video capture from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open video stream from camera.")

print("Live stream active. Press 'q' in the video window to exit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            break

        # Preprocess frame: Convert BGR to RGB and resize to model requirements
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (input_w, input_h))

        # Format input tensor based on model datatype (uint8 or float32)
        if input_dtype == np.uint8:
            input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Extract and flatten the output embedding tensor
        live_embedding = interpreter.get_tensor(output_details[0]['index']).flatten().astype(np.float32)

        # Normalize the live embedding vector
        norm = np.linalg.norm(live_embedding)
        if norm > 0:
            live_embedding = live_embedding / norm

        # Compute cosine similarity score against the target reference vector
        similarity = np.dot(ref_embedding, live_embedding)

        # Print match status to terminal without adding visual overlays
        if similarity >= SIMILARITY_THRESHOLD:
            print(f"Target Match! Similarity: {similarity:.2f}")
        else:
            print(f"Unknown. Similarity: {similarity:.2f}")

        # Display clean video output feed without any boxes or text
        cv2.imshow('Coral Edge TPU Live Recognition', frame)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed successfully.")
