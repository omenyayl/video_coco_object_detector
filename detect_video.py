import argparse
import json
import os
import sys

# import miscellaneous modules
import cv2
import numpy as np
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
import pandas as pd
from keras_retinanet.utils.visualization import draw_box, draw_caption

LABELS_T0_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush'}

MODEL = None
BOX_AREA_RATIO_TO_IGNORE = 0.2
CONFIDENCE_THRESHOLD = 0.4
MAX_CLASS_ID = 20


def main():
    global MODEL
    parser = argparse.ArgumentParser(description='This program receives a video as an input in order to look through it'
                                                 ' and process every Nth frame to detect objects.')
    parser.add_argument('--model', '-m', help='The path of the RetinaNet model to use for inference.', required=True)
    parser.add_argument('--video', '-v', help='The path to the video to process.', required=True)
    parser.add_argument('--output', '-o', help='Output directory', required=True)
    parser.add_argument('--n-frames', '-n', help='Process every N\'th frame.', default=1, type=int)
    args = parser.parse_args()
    MODEL = models.load_model(args.model, backbone_name='resnet50')
    process_video(args.video, args.n_frames, args.output)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def process_frame(frame):
    height, width, channels = frame.shape
    frame_area = width * height

    draw = frame.copy()

    frame = preprocess_image(frame)
    frame, scale = resize_image(frame)

    boxes, scores, labels, = MODEL.predict_on_batch(np.expand_dims(frame, axis=0))
    boxes /= scale

    box_json_array = []

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        b = box.astype(int)
        box_area = np.abs(b[2] - b[0]) * np.abs(b[3] - b[1])
        if score < CONFIDENCE_THRESHOLD or box_area > (frame_area * BOX_AREA_RATIO_TO_IGNORE) or label > MAX_CLASS_ID:
            break
        color = label_color(label)
        draw_box(draw, b, color=color)
        caption = '{} {:.3f}'.format(LABELS_T0_NAMES[label], score)
        draw_caption(draw, b, caption)

        box_json_array.append({
            'label': caption,
            'topLeft': [int(b[0]), int(b[1])],
            'bottomRight': [int(b[2]), int(b[3])]
        })

    return draw, box_json_array


def write_dict_as_json(filename, dict_obj):
    json_obj = json.dumps(dict_obj)
    with open(filename, 'w') as f:
        f.write(json_obj)


def display_img(img):
    cv2.imshow('Video Frame, press ESC to exit', img)
    key = cv2.waitKey(33)
    if key & 0xff == 27:
        return False
    return True


def process_video(video, n_frames, output):
    camera = cv2.VideoCapture(video)
    success, frame = camera.read()

    if not success:
        print(f'{video} is not found or it cannot be read.')
        sys.exit(1)

    count = 0
    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        if count % n_frames == 0:
            processed_frame, box_json_array = process_frame(frame)

            # Show processed frame
            # if not display_img(processed_frame):
            #     break

            base_filename = 'frame{:05d}'.format(count)
            cv2.imwrite(os.path.join(output, f'{base_filename}.jpg'), processed_frame)
            write_dict_as_json(os.path.join(output, f'{base_filename}.json'), box_json_array)

        print(f'Processed {count}/{total_frames} frames from {video}')
        count += 1

    camera.release()
    cv2.destroyAllWindows()
    print(f'Saved the results to {output}')


if __name__ == "__main__":
    main()
