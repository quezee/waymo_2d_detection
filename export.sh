python yolov5/export.py \
    --weights runs/train/model03/weights/best.pt \
    --imgsz 1920 \
    --device 0 \
    --conf-thres 0.454 \
    --iou-thres 0.6 \
    --include torchscript
