import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration constants
MODEL_PATH = 'Mobilefacenet-TF2-coral_tpu/pretrained_model/edgetpu_v1/mobilefacenet_edgetpu.tflite'
OUTPUT_EMBEDDING_PATH = 'reference_embedding.npy'
NUM_SAMPLES = 10  # Number of frames to capture and average for stability

# 1. Initialize the Edge TPU interpreter
print("Initializing Edge TPU interpreter for enrollment...")
delegate = load_delegate('libedgetpu.so.1')
interpreter = Interpreter(MODEL_PATH, experimental_delegates=[delegate])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_h, input_w = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]['dtype']

# 2. Open camera stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open video stream from camera.")

print("\n--- Face Enrollment Mode ---")
print(f"Position the target in front of the camera.")
print(f"Press 'c' to capture {NUM_SAMPLES} sequential samples, or 'q' to quit.\n")

embeddings_list = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            break

        # Show live feed with instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 'c' to Capture | 'q' to Quit", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Face Enrollment', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Enrollment cancelled.")
            break
        elif key == ord('c'):
            print(f"Capturing {NUM_SAMPLES} frames... Please hold still.")
            
            collected_count = 0
            while collected_count < NUM_SAMPLES:
                ret, sample_frame = cap.read()
                if not ret:
                    continue

                # Preprocess frame
                rgb_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_frame, (input_w, input_h))

                if input_dtype == np.uint8:
                    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
                else:
                    input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Extract and flatten embedding
                embedding = interpreter.get_tensor(output_details[0]['index']).flatten().astype(np.float32)
                
                # Normalize individual sample vector
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings_list.append(embedding)
                collected_count += 1
                
                print(f"Captured sample {collected_count}/{NUM_SAMPLES}")
                cv2.waitKey(100) # Small delay between captures

            # Compute the average embedding vector across all captured samples
            mean_embedding = np.mean(embeddings_list, axis=0)
            
            # Normalize the final mean vector
            mean_norm = np.linalg.norm(mean_embedding)
            if mean_norm > 0:
                mean_embedding = mean_embedding / mean_norm

            # Save to disk
            np.save(OUTPUT_EMBEDDING_PATH, mean_embedding)
            print(f"\nSuccess! Enrolled composite profile saved to {OUTPUT_EMBEDDING_PATH}")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Enrollment stream closed.")