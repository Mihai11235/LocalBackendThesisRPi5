import argparse
import numpy as np
import time
from PIL import Image
import cv2
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(input, height, width):
    # Load full-res image for overlay
    original_img = Image.open(input).convert('RGB')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the TFLite model.')
    parser.add_argument('--input', required=True, help='Path to the input image.')
    parser.add_argument('--output', default='overlay_result.jpg', help='Output image path.')
    args = parser.parse_args()

    # Load model
    interpreter = make_interpreter(args.model, device=':0')
    interpreter.allocate_tensors()

    # Input size from model
    width, height = common.input_size(interpreter)
    print(width, height)

    original_array, input_data = preprocess(args.input, height, width)

    # Quantize input
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details['quantization']
    input_quant = (input_data / scale + zero_point).astype(np.uint8)
    common.set_input(interpreter, input_quant)

    # Output details for dequantization
    output_details = interpreter.get_output_details()[0]
    out_scale, out_zero = output_details['quantization']

    print("🧪 Running inference...\n")

    # for i in range(5):
    start = time.perf_counter()
    interpreter.invoke()
    elapsed = (time.perf_counter() - start) * 1000  # ms

    # Dequantize and sigmoid
    output = segment.get_output(interpreter).astype(np.float32)
    output = out_scale * (output - out_zero)

    mask_bin = postprocess(original_array, output)
    result = add_green_overlay(original_array, mask_bin)

    # Save only the final output
    Image.fromarray(result).save(args.output)

    print(f"⏱️ Inference {1}: {elapsed:.2f} ms")

    print(f"\n✅ Done. Final result saved to: {args.output}")

if __name__ == '__main__':
    main()
