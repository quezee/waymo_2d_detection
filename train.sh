python yolov5/train.py \
    --weights yolov5/pretrained-models/yolov5l.pt \
    --data data/dataset.yaml \
    --hyp hyp.yaml \
    --epochs 21 \
    --batch-size 8 \
    --imgsz 1280 \
    --device 0 \
    --workers 4 \
    --save-period 2 \
    --freeze 10 \
    --project runs/train \
    --name model03