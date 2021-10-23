python /workspace/yolov5/val.py \
    --data data/dataset.yaml \
    --weights runs/train/model01/weights/best.pt \
    --batch-size 12 \
    --imgsz 1920 \
    --conf-thres 0.332 \
    --iou-thres 0.6 \
    --task test \
    --device 0 \
    --verbose \
    --project /workspace/project/runs/val \
    --name model01