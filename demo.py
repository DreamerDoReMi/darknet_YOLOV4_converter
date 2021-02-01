locals()# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = '/home/dreamer/Private/ObjectDetection/yolo-series/darknet-A-version/yolov4-rongwen20201203.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))

    #===============================================
    # rh = 608.0
    # rw = 608.0
    # h, w = img.shape[:2]
    # ratio = min(rh / h, rw / w)
    #
    # re_img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    # pad_board = np.zeros([int(rh), int(rw), 3], np.uint8)
    # if w > h:
    #     pad_board[int(rh / 2 - h * ratio / 2): int(rh / 2 + h * ratio / 2), :] = re_img
    # else:
    #     pad_board[:, int(rw / 2 - w * ratio / 2):int(rw / 2 + w * ratio / 2)] = re_img
    # # pad_board = pad_board.astype(np.float32)
    # # pad_board /= 255.0
    # sized = cv2.cvtColor(pad_board, cv2.COLOR_BGR2RGB)
    # ===============================================

    # img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    # img_in = np.expand_dims(img_in, axis=0)
    # img_in /= 255.0
    #

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.03, 0.45, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = '/home/dreamer/Private/ObjectDetection/yolo-series/darknet-A-version/yolov4-rongwen20201203.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = '/home/dreamer/Private/ObjectDetection/yolo-series/darknet-A-version/yolov4-rongwen20201203.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='/home/dreamer/Private/ObjectDetection/yolo-series/darknet-A-version/yolov4-lei.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='/home/dreamer/Private/ObjectDetection/yolo-series/darknet-A-version/yolov4-lei_best.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='/home/dreamer/workspace/RongWen/data/2020-12-09-2-pro/211.147.234.112_01_202012091419240/20201209180429_2.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfile:

        # =============================================
        file_path = '/home/dreamer/workspace/RongWen/data/tmp2'
        dst = '/home/dreamer/workspace/RongWen/data/CompareDiff/darknet_cfg_output'
        # predictons = os.path.join(WORK_DIR, 'predictions.jpg')
        predictions = '/home/dreamer/Private/ObjectDetection/yolo-series/pytorch-YOLOv4/predictions.jpg'
        for file in os.listdir(file_path):
            input = os.path.join(file_path, file)
            args.imgfile = input

            output = os.path.join(dst, file)
            # os.system('cd $wk_dir')

            detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
            os.system("cp %s %s" %(predictions, output))
        # =============================================

        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
