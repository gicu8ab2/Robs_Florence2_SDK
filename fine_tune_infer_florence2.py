import io
import os
import re
import json
import argparse
import torch
import base64
import itertools
import numpy as np
import supervision as sv
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from typing import List, Dict, Any, Tuple, Generator
from peft import LoraConfig, get_peft_model
import cv2
from roboflow import Roboflow

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RF_TOKEN = "QULagcG3YwCmtSd5GLxY"
BATCH_SIZE = 6
NUM_WORKERS = 0
EPOCHS = 10
LR = 5e-6
#REVISION = 'refs/pr/6'
REVISION = '9803f52'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:18"

class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image


class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")



def annotate_image(image, results):
    detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, results, resolution_wh=image.size)
    bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    image = bounding_box_annotator.annotate(image, detections)
    image = label_annotator.annotate(image, detections)
    image.save('./outputs/test_image.jpg')


def infer_model(val_loader, task_prompt, text_input, model, processor, image):
    height, width = image.shape[:2]
    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs, answers = next(iter(val_loader))
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    labels = processor.tokenizer(
        text=answers,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False
    ).input_ids.to(DEVICE)

    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)

    # inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    # generated_ids = model.generate(
    #     input_ids=inputs["input_ids"].to(device),
    #     pixel_values=inputs["pixel_values"].to(device),
    #     max_new_tokens=1024,
    #     early_stopping=False,
    #     ndo_sample=False,
    #     num_beams=3,
    # )
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # parsed_answer = processor.post_process_generation(
    #     generated_text,
    #     task=task_prompt,
    #     image_size=(width, height),
    # )
    # return parsed_answer



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Florence-2 Demo with PEFT inference')
    parser.add_argument('--image_path', type=str,
                        default='./images/ship_and_helicopter.jpg', help='Path to the image')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model')
    parser.add_argument('--task_prompt', type=str, default='<DENSE_REGION_CAPTION>',
                        help='Task prompt for the model')
    parser.add_argument('--text_input', type=str, default=None,
                        help='Text input for the model')
    # parser.add_argument('--model_dir', default="./models/Florence-2-base-ft",
    #                     type=str, help='Local directory to load the model')
    parser.add_argument('--model_dir', type=str, default="./model_checkpoints/epoch_9",
                        help='Output directory to save the annotated image')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True).eval().to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    image = cv2.imread(args.image_path)

    rf = Roboflow(api_key=RF_TOKEN)
    project = rf.workspace("roboflow-jvuqo").project("poker-cards-fmjio")
    version = project.version(4)
    dataset = version.download("florence2-od")
    val_dataset = DetectionDataset(
        jsonl_file_path = f"{dataset.location}/valid/annotations.jsonl",
        image_directory_path = f"{dataset.location}/valid/"
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
        revision=REVISION
    )
    peft_model = get_peft_model(model, config)
    results = infer_model(val_loader, args.task_prompt, args.text_input, peft_model, processor, image)
    # results = infer_model(args.task_prompt, args.text_input, model, processor, image)
    annotate_image(image, results)
