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


def annotate_image(image, results):
    detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, results, resolution_wh=(image.shape[1], image.shape[0]))
    bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    image = bounding_box_annotator.annotate(image, detections)
    image = label_annotator.annotate(image, detections)
    cv2.imwrite('./outputs/test_image.jpg', image)



def run_florence2_with_scores(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"
    device = model.device
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
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
    # generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=False)[0]
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
        image_size=(image.shape[1], image.shape[0]),
    )

    return parsed_answer




if __name__ == '__main__':


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
    parser.add_argument('--task_prompt', type=str, default='<DENSE_REGION_CAPTION>',
                        help='Task prompt for the model')
    parser.add_argument('--text_input', type=str, default=None,
                        help='Text input for the model')
    parser.add_argument('--output_text', type=str, default=None,
                        help='Output text path')
    args = parser.parse_args()


    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True).eval().to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    image = cv2.imread(args.image_path)
    results = run_florence2_with_scores(args.task_prompt, args.text_input, model, processor, image)

    print(results)
    annotate_image(image, results)
