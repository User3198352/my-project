'''Control by Openvino'''
#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np

import openvino as ov

from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector

FORCE_STOP = False
CAMERA_USE = False


def thread_cam1(q):
    '''Funtion detect OX from cam1'''

    # MotionDetector
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')
    # Load and initialize OpenVINO
    model_path = 'resources/openvino.xml'
    device_name = "CPU"

    # Step 1. Initialize OpenVINO Runtime Core
    core = ov.Core()

    # Step 2. Read a model
    model = core.read_model(model_path)

    ppp = ov.preprocess.PrePostProcessor(model)

    # Step 3. Set up input_tensor
    # 모델을 로드 및 장치에 적치하기 위해 inferencing에 사용할 프레임과 동일한 크기의 사진 가져오기
    input_example = cv2.imread("input_example.jpg")

    input_tensor = np.expand_dims(input_example, 0)

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - reuse precision and shape from already available `input_tensor`
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_shape(input_tensor.shape) \
        .set_element_type(ov.Type.u8) \
        .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400

    # 2) Adding explicit preprocessing steps:
    # - apply linear resize from tensor spatial dims to model spatial dims
    ppp.input().preprocess() \
        .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)

    # 3) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout(ov.Layout('NCHW'))

    # 4) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(ov.Type.f32)

    # 5) Apply preprocessing modifying the original 'model'
    model = ppp.build()

    # Step 5. Loading model to the device
    compiled_model = core.compile_model(model, device_name)

    # Open video clip resources/conveyor.mp4 instead of camera device.
    if CAMERA_USE is False:
        cap = cv2.VideoCapture('resources/conveyor.mp4')
    else:
        cap = cv2.VideoCapture(0)

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        # q.put(('Cam1 live', frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue
        q.put(('Cam1 detected', detected))

        # Enqueue "VIDEO:Cam1 detected", detected info.
        # Step 6. Set up input
        input_tensor = np.expand_dims(detected, 0)

        # Step 7. Create infer request and do inference synchronously
        results = compiled_model.infer_new_request({0: input_tensor})

        predictions = next(iter(results.values()))

        # Calculate ratios
        x_ratio = predictions[0][0]
        x_ratio = x_ratio * 100
        circle_ratio = predictions[0][1]
        circle_ratio = circle_ratio * 100
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # in queue for moving the actuator 1
        if predictions[0][0] > 0.5:
            q.put(('PUSH', 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    '''Funtion detect color from cam2'''
    # MotionDetector
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')

    # ColorDetector
    det_c = ColorDetector()
    det_c.load_preset('resources/color.cfg', 'default')

    # Open "resources/conveyor.mp4" video clip
    if CAMERA_USE is False:
        cap = cv2.VideoCapture('resources/conveyor.mp4')
    else:
        cap = cv2.VideoCapture(2)

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        # q.put(('Cam2 live', frame))

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue
        q.put(('Cam2 detected', detected))

        # Enqueue "VIDEO:Cam2 detected", detected info.
        predict = det_c.detect(frame)

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue':
            q.put(('PUSH', 2))

    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    '''show frame Funtion'''
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    '''main Funtion'''
    global FORCE_STOP
    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # Create a Queue
    q = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue.
            try:
                # de-queue name and data
                name, data = q.get_nowait()

                # show videos
                if (name == 'Cam1 live') or (name == 'Cam2 live') or (name == 'Cam1 detected') or (name == 'Cam2 detected'):
                    imshow(name, data)

                # Control actuator, name == 'PUSH's
                elif name == 'PUSH':
                    ctrl.push_actuator(data)
                else:
                    FORCE_STOP = True
                q.task_done()
            except Empty:
                pass

    t1.join()
    t2.join()
    cv2.destroyAllWindows()
    ctrl.system_stop()
    ctrl.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
