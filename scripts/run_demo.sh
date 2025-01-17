#! /bin/bash


# Florence-2 (Object Detection) with SAHI and class filter "person"
python demo_florence2_sahi.py \
    --image_path ./images/beach_scene.jpg \
    --task_prompt "object_detection" \
    --slice_wh 200 200 \
    --overlap_ratio 0.05 0.05 \
    --class_filter "person"
