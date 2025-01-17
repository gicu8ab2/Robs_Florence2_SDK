import io
import os
import argparse
import torch
import numpy as np
import supervision as sv
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
import time
import cv2
import tempfile

#TODO:  Update/replace this to map Florence-2 class label names to target ontology names
#TODO: replace class_filter to expand to target_ontology
def filter_dets(sliced_detections, area_threshold, class_filter=None):
    '''
    Keep only bounding boxes with area less than area_threshold and
    of object class class_filter if specified
    '''
    filtered_detections = sliced_detections

    locs = sliced_detections.xyxy
    areas = (locs[:,2]-locs[:,0]) * (locs[:,3]-locs[:,1])
    area_indices = np.where(areas < area_threshold)[0]

    class_names = sliced_detections.data['class_name']
    if class_filter is not None:
        class_indices = np.where(np.array(class_names) == class_filter)[0]
    else:
        class_indices = np.arange(len(class_names))

    indices = np.intersect1d(area_indices, class_indices)
    filtered_detections.xyxy = locs[indices]
    filtered_detections.data['class_name'] = [class_names[indices[i]] for i in range(len(indices))]

    return filtered_detections

def run_florence2_sahi(task_prompt, text_input, model, processor, image, slice_wh, overlap_ratio):
    device = model.device
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    def callback(image_slice: np.ndarray) -> sv.Detections:
        inputs = processor(text=prompt, images=image_slice, return_tensors="pt").to(device)
        generated_ids = model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=1024,
          early_stopping=False,
          do_sample=False,
          num_beams=3,
          return_dict_in_generate=True,
          output_scores=True,
        )

        prediction, scores, beam_indices = generated_ids.sequences, generated_ids.scores, generated_ids.beam_indices
        transition_beam_scores = model.compute_transition_scores(
            sequences=prediction,
            scores=scores,
            beam_indices=beam_indices,
        )

        parsed_answer = processor.post_process_generation(
            sequence=generated_ids.sequences[0],
            transition_beam_score=transition_beam_scores[0],
            task=task_prompt,
            image_size=slice_wh,
        )

        detection_results = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, parsed_answer, resolution_wh=slice_wh)
        return detection_results

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=slice_wh,
        overlap_ratio_wh=overlap_ratio,
        iou_threshold=0.5,
        overlap_filter=sv.OverlapFilter.NONE
        )

    sliced_detections = slicer(image=image)
    return sliced_detections



def annotate_image(image, detection_results):
    bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    image = bounding_box_annotator.annotate(image, detection_results)
    image = label_annotator.annotate(image, detection_results)
    cv2.imwrite('./outputs/test_image.jpg', image)



if __name__ == '__main__':

    TASK_PROMPT = {
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "object_detection": "<OD>",
        "dense_region_caption": "<DENSE_REGION_CAPTION>",
        "region_proposal": "<REGION_PROPOSAL>",
        "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
        "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
        "region_to_category": "<REGION_TO_CATEGORY>",
        "region_to_description": "<REGION_TO_DESCRIPTION>",
        "ocr": "<OCR>",
        "ocr_with_region": "<OCR_WITH_REGION>",
    }

    parser = argparse.ArgumentParser(description='Florence-2 Demo')
    parser.add_argument('--image_path', type=str,
                        default='./images/ship_and_helicopter.jpg', help='Path to the image')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model')
    parser.add_argument('--model_dir', default="./models/Florence-2-large",
                        type=str, help='Local directory to load the model')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory to save the annotated image')
    parser.add_argument('--task_prompt', type=str, default='dense_region_caption',
                        help='Task prompt for the model')
    parser.add_argument('--text_input', type=str, default=None,
                        help='Text input for the model')
    parser.add_argument('--output_text', type=str, default=None,
                        help='Output text path')

    #TODO: make slice_wh default to image size and overlap_ratio default to zero (for no SAHI case)
    parser.add_argument('--slice_wh', type=int, nargs=2, default=(200,200), help='Slice width and height')
    parser.add_argument('--overlap_ratio', type=float, nargs=2, default=(0.05, 0.05), help='Overlap ratio')

    parser.add_argument('--area_threshold', type=int, default=9000, help='Area threshold for filtering bboxes')
    parser.add_argument('--class_filter', type=str, default=None, help='Class filter for filtering bboxes')

    args = parser.parse_args()


    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True).eval().to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    # Load image using OpenCV
    image = cv2.imread(args.image_path)
    sliced_detections = run_florence2_sahi(TASK_PROMPT[args.task_prompt], args.text_input, model, processor,
                                        image, args.slice_wh, args.overlap_ratio)
    filtered_detections = filter_dets(sliced_detections, args.area_threshold, args.class_filter)

    print(filtered_detections)
    annotate_image(image, filtered_detections)



    #NOTE:
    florence2_to_hmie = {
        "boat": "Surface Vessel",
        "submarine": "Surface Vessel",
        "sailboat": "Surface Vessel",
        "whale": "Surface Vessel"
    }
    detected_classes = ["boat", "submarine", "sailboat", "whale"]

    mapped_classes = [florence_to_target_mapping.get(cls, "unknown") for cls in detected_classes]
