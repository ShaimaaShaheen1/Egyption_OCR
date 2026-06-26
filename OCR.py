import os
import glob
import re
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from pathlib import Path
from detectron import ArabicDigitRecognizer
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import os

# Define the absolute path where Docker will mount the models
MODEL_DIR = "/app/models"

# --- TrOCR Setup ---
checkpoint_path = os.path.join(MODEL_DIR, "TROCRcheckpoint_300_final_version")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading TrOCR on: {device}")
processor = TrOCRProcessor.from_pretrained(checkpoint_path)
trocr_model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(device)

if device.type == "cuda":
    trocr_model.half()
    print("Model converted to Half Precision (FP16)")

USE_HALF = device.type == "cuda"
YOLO_DEVICE = 0 if device.type == "cuda" else "cpu"

# --- YOLO and Custom Models Setup ---
CardLocalizationModel = YOLO(os.path.join(MODEL_DIR, "CardLocalizationModel.pt"))  
SubPartsSegmentionModel = YOLO(os.path.join(MODEL_DIR, "SubPartsSegmentionModel.pt"))
single_number_segmention_model = YOLO(os.path.join(MODEL_DIR, "best_seg_number_final.pt"))
single_name_segmention_model = YOLO(os.path.join(MODEL_DIR, "Single_name_segmentaion_model_L.pt"))

arabic_digit_model_recognizer = ArabicDigitRecognizer(num_classes=10)
arabic_digit_model_recognizer.load_model(os.path.join(MODEL_DIR, "arabic_digit_model3.pth"))

template_logo = cv2.imread(os.path.join(MODEL_DIR, 'template.jpg'), cv2.IMREAD_GRAYSCALE)
output_dir = "OUTPUT_mohamedsalah"
os.makedirs(output_dir, exist_ok=True)

# ... [Keep your crop_images, make_square, detect_card, align_id, segment_parts, single_number_segmention, Reconize_Arabic_numbers, Recognize_Arabic_Names functions as they were] ...

def crop_images(image, x_center, y_center, width, height):
    x_min, y_min = int(x_center - (width / 2)), int(y_center - (height / 2))
    x_max, y_max = int(x_center + (width / 2)), int(y_center + (height / 2))
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)
    return image[y_min:y_max, x_min:x_max]

def make_square(image):
    height, width = image.shape[:2]
    if height > width:
        pad_size = ((height - width) // 2) + 100
        padded_image = cv2.copyMakeBorder(image, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif width > height:
        pad_size = ((width - height) // 2) + 100
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded_image = image
    return padded_image

def detect_card(image_path, model):
    image = cv2.imread(image_path)
    results = model.predict(source=image_path, save=False, imgsz=640, conf=0.40,
                             half=USE_HALF, device=YOLO_DEVICE)
    return results, image

def align_id(results, image, template_logo):
    aligned_id_lst = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x_center, y_center, width, height = box.xywh[0].tolist()
            cropped_object_gray = crop_images(image, x_center, y_center, width, height)
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(template_logo, None)
            keypoints2, descriptors2 = sift.detectAndCompute(cropped_object_gray, None)
            matcher = cv2.FlannBasedMatcher()
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            if len(good_matches) > 5:
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                homography, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                rotation_angle = np.degrees(np.arctan2(homography[1, 0], homography[0, 0]))
                cropped_object_gray = make_square(cropped_object_gray)
                h, w = cropped_object_gray.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                aligned_id = cv2.warpAffine(cropped_object_gray, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                aligned_id_lst.append(aligned_id)
    return aligned_id_lst

def segment_parts(aligned_id_lst, model, file_Path):
    name, id_ = None, None
    for aligned_id in aligned_id_lst:
        results = model.predict(aligned_id, imgsz=640, conf=0.25,
                                 half=USE_HALF, device=YOLO_DEVICE)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x_center, y_center, width, height = box.xywh[0].tolist()
                class_name = model.names[int(box.cls[0])]
                if class_name == 'Name':
                    name = crop_images(aligned_id, x_center, y_center, width, height)
                elif class_name == 'ID_F':
                    id_ = crop_images(aligned_id, x_center, y_center, width+50, height)
        if name is not None:
            name = cv2.resize(name, (1000, 600))
    return name, id_

def single_number_segmention(id_region, file_Path):
    if id_region is None: return []
    results = single_number_segmention_model.predict(id_region, save=False, conf=0.01,
                                                       half=USE_HALF, device=YOLO_DEVICE)
    detected_objects = []
    for result in results:
        image = result.orig_img.copy()
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            cropped_object = image[y_min:y_max, x_min:x_max]
            detected_objects.append((x_min, x_max-x_min, cropped_object))
    
    detected_objects.sort(key=lambda x: x[0])
    filtered_objects = []
    prev_x = None
    prev_w = None
    for x_min, width, obj in detected_objects:
        if prev_x is not None and abs(x_min - prev_x) < 10:
            if width > prev_w:
                filtered_objects.pop()
                filtered_objects.append((x_min, width, obj))
        else:
            filtered_objects.append((x_min, width, obj))
        prev_x, prev_w = x_min, width
    
    final_nums = [o[2] for o in filtered_objects]
    return final_nums[:14] if len(final_nums) > 14 else final_nums

def Reconize_Arabic_numbers(all_numbers, target_size=(28, 28)):
    ID_predicted = []
    if not all_numbers: return None
    for single_number in all_numbers:
        try:
            image = cv2.cvtColor(single_number, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, target_size).astype('float32') / 255.0
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            pred, _ = arabic_digit_model_recognizer.predict(image)
            ID_predicted.append(str(pred))
        except: continue
    return "".join(ID_predicted) if ID_predicted else None

def Recognize_Arabic_Names(aligned_Card, file_path):
    if aligned_Card is None: return []
    results = single_name_segmention_model.predict(aligned_Card, save=False, conf=0.2,
                                                     half=USE_HALF, device=YOLO_DEVICE)
    objs = []
    for result in results:
        img = result.orig_img.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objs.append([img[y1:y2, x1:x2], x1, y1, x2, y2, (x1+x2)/2, (y1+y2)/2, (x2-x1)*(y2-y1)])
    # Simplified filtering for clarity
    objs.sort(key=lambda x: x[7], reverse=True)
    filtered = []
    for o in objs:
        keep = True
        for f in filtered:
            if o[1] >= f[1] and o[3] <= f[3] and o[2] >= f[2] and o[4] <= f[4]:
                keep = False; break
        if keep: filtered.append(o)
    return filtered

# --- Updated OCR Function with Batching and TrOCR ---
def Read_arabic_names(detected_objects, batch_size=8):
    if not detected_objects:
        return []

    # 1. Row Sorting Logic
    detected_objects.sort(key=lambda x: x[2])  # Sort by y_min
    all_sorted_objs = []
    
    first_row = [detected_objects[0]]
    second_row = detected_objects[1:]
    
    # Sort rows Right-to-Left (Arabic)
    first_row.sort(key=lambda x: x[1], reverse=True)
    second_row.sort(key=lambda x: x[1], reverse=True)
    
    all_sorted_objs.extend(first_row)
    all_sorted_objs.extend(second_row)

    # 2. Extract images and convert to PIL RGB
    pil_images = []
    for obj in all_sorted_objs:
        cv2_img = cv2.cvtColor(obj[0], cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(cv2_img))

    predicted_names = []

    # 3. Batch Inference
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        
        # Preprocessing
        inputs = processor(images=batch, return_tensors="pt").to(device)
        
        # Apply half precision to inputs if on GPU
        if device.type == 'cuda':
            inputs.pixel_values = inputs.pixel_values.to(torch.float16)

        # Generation
        with torch.no_grad():
            generated_ids = trocr_model.generate(inputs.pixel_values)
        
        # Decoding
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predicted_names.extend([text.strip() for text in generated_text])

    for name in predicted_names:
        print(f"الاسم المكتشف: {name}")

    return predicted_names

import time

def process_image(image_path):
    print(f"\n--- Processing Image: {Path(image_path).name} ---")
    total_start = time.time()

    # 1. Card Detection
    start = time.time()
    results, image = detect_card(image_path, CardLocalizationModel)
    print(f"[*] Card Detection: {time.time() - start:.3f}s")

    # 2. Alignment (SIFT)
    start = time.time()
    aligned_id_lst = align_id(results, image, template_logo)
    print(f"[*] Image Alignment (SIFT): {time.time() - start:.3f}s")

    if not aligned_id_lst:
        print("[!] Error: No card aligned. Skipping...")
        return [[], None]

    # 3. Sub-Parts Segmentation (Name/ID Regions)
    start = time.time()
    name_region, id_region = segment_parts(aligned_id_lst, SubPartsSegmentionModel, image_path)
    print(f"[*] Sub-Parts Segmentation: {time.time() - start:.3f}s")

    # 4. Name Parts Segmentation (YOLO-L)
    start = time.time()
    names_parts = Recognize_Arabic_Names(aligned_id_lst[0], image_path)
    print(f"[*] Name Parts Segmentation (YOLO-L): {time.time() - start:.3f}s")

    # 5. Name Recognition (Tesseract)
    start = time.time()
    Name = Read_arabic_names(names_parts)
    print(f"[*] Name OCR (Tesseract): {time.time() - start:.3f}s")

    # 6. Digit Segmentation
    start = time.time()
    output_cropped_numbers = single_number_segmention(id_region, image_path)
    print(f"[*] Digit Segmentation: {time.time() - start:.3f}s")

    # 7. Digit Recognition (CNN)
    start = time.time()
    ID_number = Reconize_Arabic_numbers(output_cropped_numbers)
    print(f"[*] Digit Recognition: {time.time() - start:.3f}s")

    total_duration = time.time() - total_start
    print(f"--- TOTAL TIME: {total_duration:.3f}s ---\n")

    return [Name, ID_number]

def func(image_path):
    try:
        predicted_id = process_image(image_path)
        names = predicted_id[0]
        if not names: return ["error", "error", "error"]
        
        first_name = names[0]
        final_name = " ".join(names[1:])
        id_str = str(predicted_id[1])
        if len(id_str) != 14:
            id_str = "id is not 14 digits"
        return [id_str, first_name, final_name]
    except Exception as e:
        print(f"General Error: {e}")
        return ["error", "error", "error"]

