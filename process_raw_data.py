import argparse
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader.dataset_pb2 import \
    CameraName, CameraImage, CameraCalibration, CameraLabels
from google.protobuf.pyext._message import RepeatedCompositeContainer


def filter_container(container: RepeatedCompositeContainer,
                     camera_name: int = CameraName.FRONT) -> \
                Union[CameraImage, CameraCalibration, CameraLabels]:
    '''
    Extracts data from RepeatedCompositeContainer according to given
    camera_name.
    '''
    container_filt = filter(lambda x: x.name == camera_name, container)
    return next(container_filt)

def process_waymo_file(source_file: Path, dest_dir: Path,
                       label_map: dict) -> List[Tuple]:
    '''
    Dumps each frame of Waymo TFR file as .jpg and .txt
    (for annotations), also returning descriptive table for EDA.
    '''
    data = []
    datafile = WaymoDataFileReader(source_file)
    source_file_id = re.findall('-([0-9_]+)_w', source_file.name)[0]
    (dest_dir / 'images' / source_file_id).mkdir(parents=True, exist_ok=True)
    (dest_dir / 'labels' / source_file_id).mkdir(parents=True, exist_ok=True)

    for frame_idx, frame in enumerate(datafile):
        annotations = filter_container(frame.camera_labels)
        if not annotations.labels:
            continue
        img = filter_container(frame.images)
        calibration = filter_container(frame.context.camera_calibrations)
        
        width = calibration.width
        height = calibration.height

        image_id = source_file_id + '/' + str(frame_idx)
        image_path = dest_dir / 'images' / source_file_id / f'{frame_idx}.jpg'
        with open(image_path, 'wb') as fid:
            fid.write(img.image)

        annot_path = dest_dir / 'labels' / source_file_id / f'{frame_idx}.txt'
        annot_fid = open(annot_path, 'w')

        for annot in annotations.labels:
            x = annot.box.center_x / width
            y = annot.box.center_y / height
            w = annot.box.length / width
            h = annot.box.width / height
            scale = np.sqrt(w * h)
            aspect_ratio = w / h
            label = label_map[annot.type]

            annot_line = ' '.join(map(str, [label, x, y, w, h]))
            annot_fid.write(annot_line + '\n')
            
            data.append((
                str(image_path), image_id, height, width,
                label, x, y, w, h, scale, aspect_ratio
            ))

        annot_fid.close()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Waymo dataset to YOLOv5 compatible format'
    )
    parser.add_argument('--source_dir', default='/mnt/data',
                        help='path to Waymo OD directory')
    parser.add_argument('--dest_dir', default='/workspace/project/data',
                        help='where to store processed data')
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    label_map = {1: 0, 2: 1, 4: 2}
    data = []

    for source_file in source_dir.glob('*.tfrecord'):
        data.extend(
            process_waymo_file(source_file, dest_dir, label_map)
        )

    columns = ['image_path', 'image_id', 'height', 'width', 'category_id',
               'xmin', 'ymin', 'xmax', 'ymax', 'scale', 'aspect_ratio']
    df = pd.DataFrame(data=data, columns=columns)
    df.to_pickle(dest_dir / 'df_annotations.pkl')

    image_paths = df.image_path.unique()
    with open(dest_dir / 'splits/all_imgs.txt', 'w') as fid:
        fid.write('\n'.join(image_paths))
