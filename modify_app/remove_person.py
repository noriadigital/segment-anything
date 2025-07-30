import cv2
import numpy as np
import torch
import clip
import sys
import os

# Agregamos el directorio padre al path (donde está `segment_anything`)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

def choose_mask_clip(input_image, input_masks, output_path='', return_image=False, object= ''):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Multiple clothing-specific prompts
    if not object:
        clothing_prompts = [
            "a photo of a person wearing clothes",
            "a selfie of a single person with clothing garments",
            "a person standing with clothes",
            "an individual in a selfie showing off clothes",
            "a person smiling in a selfie",
            "a selfie taken with a smartphone, with a person wearing clothes"
        ]
    else:
        clothing_prompts = [
            object,
            f"a close-up photo of a {object}",
            f"a single piece of clothing," + object,
            "clothing product photography" + object,
            "retail clothing item" + object,
            "fashion item photography" + object
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

def remove_background_to_white(image_path, output_path, garment,
                               sam_checkpoint="../sam_vit_h_4b8939.pth",
                               model_type="vit_h",
                               points_per_side=16,
                               pred_iou_thresh=0.88,
                               stability_score_thresh=0.90,
                               min_mask_region_area=1000):

    # 1. Leer imagen
    image = cv2.imread(image_path)
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

    # 4. Elegir máscara usando clip (elegimos la persona)
    mask = choose_mask_clip(image,masks) # array booleana HxW

    # 5. Aplicar máscara sobre imagen
    masked = image.copy()
    masked[~mask] = [255, 255, 255]  # fondo blanco

    result = cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)
    cv2.imwrite("process.jpg", result)

    #6. Volver a segmentar imagen recortada
    masks2 = mask_generator.generate(result)

    #7. Elegir máscara usando clip (elegimos la ropa)

    ## 7.1 Armamos prompts para ropa
    prompt = 'A single clothing item, such as a ' + garment

    ## 7.2 Lo pasamos por clip
    mask2 = choose_mask_clip(result, masks2, object = prompt)

    # 8. Guardar imagen
    masked2 = result.copy()
    masked2[~mask2] = [255, 255, 255]  # fondo blanco

    cv2.imwrite(output_path, masked2)
    print(f"Imagen guardada en: {output_path}")
    return True


remove_background_to_white('../notebooks/images/selfie2.jpg', 'prueb3.jpg', "dress",
                               sam_checkpoint="../sam_vit_h_4b8939.pth",
                               model_type="vit_h",
                               points_per_side=16,
                               pred_iou_thresh=0.88,
                               stability_score_thresh=0.90,
                               min_mask_region_area=1000)