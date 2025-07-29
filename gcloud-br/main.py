import cv2
import base64
import numpy as np
import torch
import clip
import sys
import os
from google.cloud import storage
import functions_framework
from flask import Request, jsonify
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def choose_mask_clip(text_prompt, input_image, input_masks, output_path='', return_image=False):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Multiple clothing-specific prompts
    clothing_prompts = [
        "a clothing item on plain background" + text_prompt,
        "a single piece of clothing" + text_prompt,
        "clothing product photography" + text_prompt,
        "retail clothing item" + text_prompt,
        "fashion item photography" + text_prompt
    ]
    text_tokens = clip.tokenize(clothing_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    mask_shape = input_masks[0]["segmentation"].shape
    image_area = mask_shape[0] * mask_shape[1]
    best_score = -1
    best_mask = None

    for i, mask in enumerate(input_masks):
        segmentation = mask["segmentation"]
        
        # Skip masks that are too large (likely background)
        mask_area = np.sum(segmentation)
        if mask_area / image_area > 0.9:  # Skip if mask covers >90% of image
            continue

        # Apply mask to original image
        masked_image_np = input_image.copy()
        #Todo lo que no es segmentation es igual a 0.
        masked_image_np[~segmentation] = 0

        # Preprocess for CLIP
        masked_pil = Image.fromarray(masked_image_np)
        processed = preprocess(masked_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(processed)
            # Calculate similarity with all prompts
            similarities = torch.cosine_similarity(image_features, text_features)
            max_similarity = torch.max(similarities).item()
            
            # Add penalty for very large or very small masks
            area_ratio = mask_area / image_area
            size_penalty = 1.0
            if area_ratio > 0.7 or area_ratio < 0.05:
                size_penalty = 0.5
            
            final_score = max_similarity * size_penalty

        if final_score > best_score:
            best_mask = segmentation
            best_score = final_score

    if return_image == False:
        return best_mask

    # Create output image with alpha channel
    masked_img = cv2.bitwise_and(input_image, input_image, mask=best_mask.astype(np.uint8))
    b, g, r = cv2.split(masked_img)
    alpha = best_mask.astype(np.uint8) * 255
    result = cv2.merge((b, g, r, alpha))
    result_bgra = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)

    if output_path:
        cv2.imwrite(os.path.join(output_path, f"mask_{i+1:03}.png"), result_bgra)
        return "Image saved in folder"
    return result_bgra

@functions_framework.http
def remove_background_to_white_handler(request: Request):
    data = request.get_json()
    image_b64 = data.get("image_base64")
    garment = data.get("garment")
    sam_checkpoint = download_blob(
        bucket_name="modify-assets",
        source_blob_name="modify-assets/sam_checkpoints/sam_vit_h_4b8939.pth",
        destination_file_name="sam_checkpoint.pth"
    )
    model_type = data.get("model_type", "vit_h")
    points_per_side = int(data.get("points_per_side", 16))
    pred_iou_thresh = float(data.get("pred_iou_thresh", 0.88))
    stability_score_thresh = float(data.get("stability_score_thresh", 0.90))
    min_mask_region_area = int(data.get("min_mask_region_area", 1000))

    # 1. Leer imagen
    img_bytes = base64.b64decode(image_b64)
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Cargar modelo SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area
    )

    # 3. Generar máscaras
    masks = mask_generator.generate(image)
    if not masks:
        print("No se detectaron máscaras.")
        return False

    # 4. Elegir máscara usando clip
    mask = choose_mask_clip('A single clothing item, such as a ' + garment,image,masks) # array booleana HxW

    # 5. Aplicar máscara sobre imagen
    masked = image.copy()
    masked[~mask] = [255, 255, 255]  # fondo blanco

    result = cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", result)
    res_b64 = base64.b64encode(buf).decode("utf-8")
    print("Imagen Procesada!")

    # 6. Subir imagen procesada a GCS
    output_bucket_name = "modify-assets"
    output_path = "modify-assets/output_garments"
    output_file_name = "output_prueba.jpg"

    upload_blob(
        bucket_name=output_bucket_name,
        source_file_name=output_path,
        destination_blob_name = output_file_name
    )

    return jsonify({"image_processed_base64": res_b64})