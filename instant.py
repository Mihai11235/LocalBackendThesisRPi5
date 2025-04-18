import cv2
import numpy as np
from PIL import Image
import asyncio
import threading
import traceback

import time
from PIL import Image
import cv2
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

from picamera2 import Picamera2
from picamera2.utils import Transform



picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    transform=Transform(),
    controls={"AfMode": 2}
))
picam2.start()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(input, height, width):
    # Load full-res image for overlay
    original_img = Image.fromarray(input).convert('RGB')
    original_array = np.array(original_img)

    # Resize to model size and normalize
    resized_img = original_img.resize((width, height), Image.LANCZOS)
    input_data = np.asarray(resized_img).astype(np.float32) / 255.0

    return original_array, input_data

def postprocess(original_array, output):
    prob = sigmoid(output)
    mask = np.squeeze(prob)  # (128, 128)

    # Resize mask back to original size
    mask_resized = cv2.resize(mask, original_array.shape[:2][::-1])  # (W,H)

    # Threshold mask
    mask_bin = (mask_resized > 0.5).astype(np.uint8)

    return mask_bin

def add_green_overlay(original_array, mask_bin):
    # Create green overlay
    green = np.zeros_like(original_array)
    green[:, :, 1] = 255
    overlay = (np.expand_dims(mask_bin, axis=2) * green).astype(np.uint8)

    # Blend overlay on original
    result = cv2.addWeighted(original_array, 1.0, overlay, 0.5, 0)

    return result




def process_frame(frame, interpreter, width, height):

    original_array, input_data = preprocess(frame, height, width)

    # Quantize input
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details['quantization']
    input_quant = (input_data / scale + zero_point).astype(np.uint8)
    common.set_input(interpreter, input_quant)

    # Output details for dequantization
    output_details = interpreter.get_output_details()[0]
    out_scale, out_zero = output_details['quantization']

    start = time.perf_counter()
    interpreter.invoke()
    elapsed = (time.perf_counter() - start) * 1000  # ms

    # Dequantize and sigmoid
    output = segment.get_output(interpreter).astype(np.float32)
    output = out_scale * (output - out_zero)

    mask_bin = postprocess(original_array, output)
    # result = add_green_overlay(original_array, mask_bin)
    green = np.zeros_like(original_array)
    green[:, :, 1] = 255
    overlay = (np.expand_dims(mask_bin, axis=2) * green).astype(np.uint8)

    # Blend overlay on original
    # result = cv2.addWeighted(original_array, 1.0, overlay, 0.5, 0)

    # return result
    return overlay




    # # Convert frame to PIL Image
    # pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # # Apply transformations
    # input_tensor = transform(pil_image).unsqueeze(0)
    
    # # Run inference
    # with torch.no_grad():
    #     output = model(input_tensor)
    
    # # Convert output to numpy array
    # mask = output.squeeze().cpu().numpy()
    
    # # Resize mask to match frame size
    # mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # # Invert the mask
    # inverted_mask = 1 - mask
    
    # # Create green overlay
    # green_overlay = np.zeros_like(frame)
    # green_overlay[:, :, 1] = 255  # Set green channel to 255
    
    # # Apply inverted mask to green overlay
    # mask_3d = np.stack([inverted_mask, inverted_mask, inverted_mask], axis=2)
    # green_mask = (mask_3d * green_overlay).astype(np.uint8)
    
    # return green_mask  # Return just the mask overlay (not blended)


async def process_frame_async(frame, interpreter, width, height):
    # Simulate async model inference
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, process_frame, frame, interpreter, width, height)


# This function handles the frame display and processing logic
def capture_and_process(interpreter, width, height):
    try:
        # Open the device camera (index 0 for default camera)
        # camera = cv2.VideoCapture(0)  # Use 1 or other indices for multiple cameras
        
        # Optionally, set camera resolution
        # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



        # Initialize a variable to store the last processed mask overlay
        last_overlay = None
        print("Starting capture loop...")
        while True:
            # ret, frame = camera.read()
            print("Capturing frame...")
            frame = picam2.capture_array()
            # if not ret:
            #     print("Failed to grab frame from camera.")
            #     break
            
            print("Processing frame...")
            # Process frame and get the mask overlay
            result_frame = asyncio.run(process_frame_async(frame, interpreter, width, height))
            print("Frame processed.")


            # if result_frame is not None:
            #     alpha = 0.9
            #     blended_frame = cv2.addWeighted(frame, 1, result_frame, alpha, 0)
            #     print("Blended frame shape:", blended_frame.shape)
            #     cv2.imwrite("debug_output.jpg", blended_frame)
            #     print("Saved to debug_output.jpg")
            #     cv2.imshow('Lane Detection with Overlay', blended_frame)
                # print("Showing frame...")
            # Blend mask overlay with original frame only if result_frame is available
            if result_frame is not None:
                print("1")
                alpha = 0.9  # Adjust transparency of the overlay
                print(2)
                blended_frame = cv2.addWeighted(frame, 1, result_frame, alpha, 0)
                print(3)
                # cv2.namedWindow("Lane Detection with Overlay", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("Lane Detection with Overlay", 1280, 720)
                print(4)
                cv2.imshow('Lane Detection with Overlay', blended_frame)
                print(5)
            else:
                print("Original frame shown!\n")
                # If no result is available, show the original frame
                cv2.imshow('Lane Detection with Overlay', frame)
            print(6)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed.")
                break

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
    finally:
        # Release resources
        # camera.release()
        cv2.destroyAllWindows()


# async def main():
#     # Load model
#     interpreter = make_interpreter("model/model_quant_edgetpu.tflite", device=':0')
#     interpreter.allocate_tensors()

#     # Input size from model
#     width, height = common.input_size(interpreter)

#     # Start the video capture and frame processing in a separate thread
#     video_thread = threading.Thread(target=capture_and_process, args=(interpreter, width, height))
#     video_thread.start()

#     video_thread.join()


async def main():
    print("Initializing interpreter...")
    interpreter = make_interpreter("model/model_quant_edgetpu.tflite", device=':0')
    interpreter.allocate_tensors()
    print("Interpreter ready.")

    width, height = common.input_size(interpreter)
    print(f"Model input size: {width}x{height}")

    video_thread = threading.Thread(target=capture_and_process, args=(interpreter, width, height))
    video_thread.start()

    video_thread.join()



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        print("Fatal error:")
        traceback.print_exc()
