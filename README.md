# Project: Object Detection in an Urban Environment

Here you can see my object detection pipeline using data from [Waymo Real-time 2D Detection](https://waymo.com/open/challenges/2021/real-time-2d-prediction/) (only the first 100 training TFR files, amounting to 94 GB).
I've tried [Tensorflow OD API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [Pytorch's Detectron2](https://github.com/facebookresearch/detectron2) but ultimately decided to go with [Ultralytics' YOLOv5](https://github.com/ultralytics/yolov5) due to its performance in terms of both accuracy and speed.

## Prerequisites

To launch training/testing pipeline you are expected to have aforementioned 100 TFR files in `/mnt/waymo_od` on your machine.
To build docker image with necessary environment, move to [/build](https://github.com/quezee/nd013c1_yolo/tree/master/build) and use these commands:
`docker build -t nd013c1_yolo -f Dockerfile.main .`
`docker build -t tensorboard -f Dockerfile.tb .`