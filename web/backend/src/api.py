
import queue
import smtplib
from common.embeddings import FN_Embeddings
from common.recognize_event import *
from common.faceinfo import *
from common.common import *
from common.enums import *
from common.settings import SettingsInfo
import tensorflow as tf
import random
import time
import cv2
import threading
import json
from flask import request, jsonify
from flask_cors import CORS
from flask_restful import Resource, Api
from flask import Response
from flask import Flask
import os
import subprocess
import numpy as np
import sys

from pathlib import Path

from datetime import datetime

# get tf to be quiet -> doesnt work yet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#import torch
#from torch.utils.data import DataLoader
# rom torchvision import datasets
#from torchvision.transforms import functional as F

#import pandas as pd
#from mtcnn.mtcnn import MTCNN

try:
    from nn.mtcnn import TrtMtcnn
except:
    print("Failed to load mtcnn")

# TODO consoludate under one import


#from facenet_pytorch import MTCNN, InceptionResnetV1

sys.path.append("/home/karlo/github/facenet")  # TODO add this to python path
print(sys.path)

#from contributed.face import *


# To speed up performance durring dev
runEncoder = True

instantRecord = True


# try:
#    from denoiser.knn import KNN_Denoiser
# except:
#    print("failed to load KNN_Denoiser")

#import face_recognition

#import imutils

q = queue.Queue()

if os.name != 'nt':
    currentSettings = SettingsInfo(
        DetectionMode.DetectFaces, VideoMode.File, "/home/karlo/vid/vid8.mp4")
else:
    currentSettings = SettingsInfo(
        DetectionMode.DetectFaces, VideoMode.File, "I:\\Maturaarbeit\\vid.mp4")


# If the settings.json file can be find read from it
currentSettings.readFromDisk()

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


allFaces = []


todos = {}

#lock = threading.Lock()

#videoFeedPath = "I:\\Maturaarbeit\\vid.mp4"
#videoFeedPath = "/home/karlo/vid/vid.mp4"

raw_image = None


isFromFile = True
resetVideo = False
outputFrame = None
lastface_img = None

encodedImage_face = None
encodedImage_face_recognized = None

# create the detector, using default weights
#detector = MTCNN()
#detector = None

detector = None

if os.name != 'nt':
    detector = TrtMtcnn()

faceEvents = []

# Facenet
FaceNetEncoder = None
if runEncoder:
    FaceNetEncoder = FN_Embeddings()

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO see to use the current mtcnn
# mtcnn = MTCNN(
#    image_size=160, margin=0, min_face_size=20,
#    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
#    device=device
# )

#resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#knn = None

# try:
#    knn = KNN_Denoiser(2520, 1080)
# except:
#    print("can't load KNN_Denoiser")

# TODO Ensure sudo permissions


def ensure_camera_ready():
    cmd = ['sudo systemctl restart nvargus-daemon']
    hostname = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]


def process_queue():
    global currentSettings
    global q

    # gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=3280, height=1884,format=NV12, framerate=21/1' ! omxh264enc ! qtmux ! filesink location=LESSON_DAY_MONTH_COUNT.avi -e

    # TODO run https://developer.ridgerun.com/wiki/index.php?title=Jetson_Nano/Gstreamer/Example_Pipelines/Encoding to record -> less cpu usage and faster

    (live_width, live_height, live_fps) = getResolutionAndFramerate(
        currentSettings.liveResolution)

    print("process_queue")
    print(currentSettings.liveResolution)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output_' + time.strftime("%Y%m%d-%H%M%S") +
                          '.avi', fourcc, float(live_fps), (live_width, live_height))

    while True:
        print("SAVE TO DISK: " + str(q.qsize()))

        if currentSettings.currentDetectionMode != DetectionMode.Record and q.empty():
            break

        if not q.empty():
            capturedframe = q.get()
            img = capturedframe["imagecaptured"]
            out.write(img)
            time.sleep(0.01)
        else:
            time.sleep(0.05)

    print('RELEASING')
    out.release()


def live_view():
    global outputFrame
    global raw_image
    global resetVideo

    # https://devtalk.nvidia.com/default/topic/1051913/jetson-nano/how-to-close-gstreamer-pipeline-in-python/post/5340291/#5340291

    # Hardcoded as this one provides the fill with view
    (live_width, live_height, live_fps) = getResolutionAndFramerate(
        LiveResolution.Live_3264_1848)

    # TODO maybe api
    flipmode = 0

    # TO Restart camera
    # sudo systemctl restart nvargus-daemon

    '''
    karlo@karlo:~$ gst-inspect-1.0 nvarguscamerasrc
Factory Details:
  Rank                     primary (256)
  Long-name                NvArgusCameraSrc
  Klass                    Video/Capture
  Description              nVidia ARGUS Camera Source
  Author                   Viranjan Pagar <vpagar@nvidia.com>, Amit Pandya <apandya@nvidia.com>

Plugin Details:
  Name                     nvarguscamerasrc
  Description              nVidia ARGUS Source Component
  Filename                 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvarguscamerasrc.so
  Version                  1.0.0
  License                  Proprietary
  Source module            nvarguscamerasrc
  Binary package           NvARGUSCameraSrc
  Origin URL               http://nvidia.com/

GObject
 +----GInitiallyUnowned
       +----GstObject
             +----GstElement
                   +----GstBaseSrc
                         +----GstNvArgusCameraSrc

Pad Templates:
  SRC template: 'src'
    Availability: Always
    Capabilities:
      video/x-raw(memory:NVMM)
                  width: [ 1, 2147483647 ]
                 height: [ 1, 2147483647 ]
                 format: { (string)NV12 }
              framerate: [ 0/1, 120/1 ]

Element has no clocking capabilities.
Element has no URI handling capabilities.

Pads:
  SRC: 'src'
    Pad Template: 'src'

Element Properties:
  name                : The name of the object
                        flags: readable, writable
                        String. Default: "nvarguscamerasrc0"
  parent              : The parent of the object
                        flags: readable, writable
                        Object of type "GstObject"
  blocksize           : Size in bytes to read per buffer (-1 = default)
                        flags: readable, writable
                        Unsigned Integer. Range: 0 - 4294967295 Default: 4096
  num-buffers         : Number of buffers to output before sending EOS (-1 = unlimited)
                        flags: readable, writable
                        Integer. Range: -1 - 2147483647 Default: -1
  typefind            : Run typefind before negotiating (deprecated, non-functional)
                        flags: readable, writable, deprecated
                        Boolean. Default: false
  do-timestamp        : Apply current stream time to buffers
                        flags: readable, writable
                        Boolean. Default: true
  silent              : Produce verbose output ?
                        flags: readable, writable
                        Boolean. Default: true
  timeout             : timeout to capture in seconds (Either specify timeout or num-buffers, not both)
                        flags: readable, writable
                        Unsigned Integer. Range: 0 - 2147483647 Default: 0
  wbmode              : White balance affects the color temperature of the photo
                        flags: readable, writable
                        Enum "GstNvArgusCamWBMode" Default: 1, "auto"
                           (0): off              - GST_NVCAM_WB_MODE_OFF
                           (1): auto             - GST_NVCAM_WB_MODE_AUTO
                           (2): incandescent     - GST_NVCAM_WB_MODE_INCANDESCENT
                           (3): fluorescent      - GST_NVCAM_WB_MODE_FLUORESCENT
                           (4): warm-fluorescent - GST_NVCAM_WB_MODE_WARM_FLUORESCENT
                           (5): daylight         - GST_NVCAM_WB_MODE_DAYLIGHT
                           (6): cloudy-daylight  - GST_NVCAM_WB_MODE_CLOUDY_DAYLIGHT
                           (7): twilight         - GST_NVCAM_WB_MODE_TWILIGHT
                           (8): shade            - GST_NVCAM_WB_MODE_SHADE
                           (9): manual           - GST_NVCAM_WB_MODE_MANUAL
  saturation          : Property to adjust saturation value
                        flags: readable, writable
                        Float. Range:               0 -               2 Default:               1
  sensor-id           : Set the id of camera sensor to use. Default 0.
                        flags: readable, writable
                        Integer. Range: 0 - 255 Default: 0
  exposuretimerange   : Property to adjust exposure time range in nanoseconds
                        Use string with values of Exposure Time Range (low, high)
                        in that order, to set the property.
                        eg: exposuretimerange="34000 358733000"
                        flags: readable, writable
                        String. Default: null
  gainrange           : Property to adjust gain range
                        Use string with values of Gain Time Range (low, high)
                        in that order, to set the property.
                        eg: gainrange="1 16"
                        flags: readable, writable
                        String. Default: null
  ispdigitalgainrange : Property to adjust digital gain range
                        Use string with values of ISP Digital Gain Range (low, high)
                        in that order, to set the property.
                        eg: ispdigitalgainrange="1 8"
                        flags: readable, writable
                        String. Default: null
  tnr-strength        : property to adjust temporal noise reduction strength
                        flags: readable, writable
                        Float. Range:              -1 -               1 Default:              -1
  tnr-mode            : property to select temporal noise reduction mode
                        flags: readable, writable
                        Enum "GstNvArgusCamTNRMode" Default: 1, "NoiseReduction_Fast"
                           (0): NoiseReduction_Off - GST_NVCAM_NR_OFF
                           (1): NoiseReduction_Fast - GST_NVCAM_NR_FAST
                           (2): NoiseReduction_HighQuality - GST_NVCAM_NR_HIGHQUALITY
  ee-mode             : property to select edge enhnacement mode
                        flags: readable, writable
                        Enum "GstNvArgusCamEEMode" Default: 1, "EdgeEnhancement_Fast"
                           (0): EdgeEnhancement_Off - GST_NVCAM_EE_OFF
                           (1): EdgeEnhancement_Fast - GST_NVCAM_EE_FAST
                           (2): EdgeEnhancement_HighQuality - GST_NVCAM_EE_HIGHQUALITY
  ee-strength         : property to adjust edge enhancement strength
                        flags: readable, writable
                        Float. Range:              -1 -               1 Default:              -1
  aeantibanding       : property to set the auto exposure antibanding mode
                        flags: readable, writable
                        Enum "GstNvArgusCamAeAntiBandingMode" Default: 1, "AeAntibandingMode_Auto"
                           (0): AeAntibandingMode_Off - GST_NVCAM_AEANTIBANDING_OFF
                           (1): AeAntibandingMode_Auto - GST_NVCAM_AEANTIBANDING_AUTO
                           (2): AeAntibandingMode_50HZ - GST_NVCAM_AEANTIBANDING_50HZ
                           (3): AeAntibandingMode_60HZ - GST_NVCAM_AEANTIBANDING_60HZ
  exposurecompensation: property to adjust exposure compensation
                        flags: readable, writable
                        Float. Range:              -2 -               2 Default:               0
  aelock              : set or unset the auto exposure lock
                        flags: readable, writable
                        Boolean. Default: false
  awblock             : set or unset the auto white balance lock
                        flags: readable, writable
                        Boolean. Default: false
  maxperf             : set or unset the max performace
                        flags: readable, writable
                        Boolean. Default: false
  bufapi-version      : set to use new Buffer API
                        flags: readable, writable
                        Boolean. Default: false

    '''





    exposure = 5000000 
    exposure = 10000000
    # exposuretimerange=\"{exposure} {exposure}\" 
    #wbmode=0 awblock=true gainrange=\"8 8\" ispdigitalgainrange=\"4 4\" exposuretimerange=\"5000000 5000000\" aelock=true 
    #-
    #tnr-mode=2 tnr-strength=1 

    device_name = f"nvarguscamerasrc ! video/x-raw(memory:NVMM), "
    device_name += f"width=(int){live_width}, height=(int){live_height},format=(string)NV12, framerate=(fraction){live_fps}/1 "
    device_name += f"! nvvidconv flip-method={flipmode} ! video/x-raw, width=(int){live_width}, height=(int){live_height}, format=(string)BGRx "
    device_name += "! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
    #device_name += "! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

    print(device_name)
    # TODO see to remove the 2 sec sleep

    #cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    cap = cv2.VideoCapture(device_name)
    if cap.isOpened():
        print("camera LIVE")
        print(time.time())
        print("camera LIVE")

        # TODO Get path to save
        # TODO Detect if recording active
        start = time.process_time()
        count = 0
        while True:
            # try:

            ret_val, img = cap.read()

            start_de = time.time()

            if ret_val == False:
                print("Frame is empty face_recognition_man live")
                time.sleep(0.5)
                continue

            raw_image = img.copy()

            orig_width = 3264
            orig_height = 1848

            tensor_rt_width = 2520
            tensor_rt_heigth = 1080

            strech_factor = orig_width / float(tensor_rt_width)
            needede_height = tensor_rt_heigth * strech_factor
            obsolete_height = orig_height - needede_height

            overlay = img.copy()

            #print("orig_width" + str(orig_width))
            #print("strech_factor" + str(strech_factor))
            #print("needede_height" + str(needede_height))
            #print("obsoleteHeight" + str(obsoleteHeight))

            cv2.rectangle(overlay, (0, 0), (orig_width, int(obsolete_height/2)), (0, 0, 255), -1)
            cv2.rectangle(overlay, (0, int((obsolete_height/2) + needede_height)), (orig_width, orig_height), (0, 255, 0), -1)

            timestamp = datetime.now()
            cv2.putText(img, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(img, str(fps), (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            time.sleep(0.01)
            # cv2.waitKey(1)

            outputFrame = img.copy()
            # to see what mtcnn sees

            y_start = int(obsolete_height/2)

            #outputFrame = img[y_start:int(y_start+neededeHeight), 0:orig_width]

            cv2.addWeighted(overlay, 0.5, outputFrame, 0.5, 0, outputFrame)

            count += 1

            if(time.time() - start > 1):
                print(str(time.time()) + " | Current FPS: " + str(count))
                count = 0
                start = time.time()

            #print(time.process_time() - start)
            if resetVideo:
                cap.release()
                

                print("BREAKING VIDEO BEACUSE RESET VIDEO IS TRUE")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("BREAKING VIDEO BEACUSE RESET VIDEO IS TRUE")

                break
            # except:
            #    print("EXCEPTION CAPTURE")
    else:
        print("camera open failed")
        # Issue reboot command

    print("EXIT LIVE VIEW")


def face_recognition_man():
    frameCount = 30

    # grab global references to the video stream, output frame, and
    # lock variables
    global vs
    global outputFrame
    global raw_image
    global resetVideo

    global currentSettings
    global q

    #global knn

    cap = None

    print('mode: ' + str(currentSettings.currentVideoMode) + ' reset: ' +
          str(resetVideo) + ' path: ' + currentSettings.videoPath)

    # TODO Switch
    if currentSettings.currentVideoMode == VideoMode.File:

        frame_counter = 0
        cap = cv2.VideoCapture(currentSettings.videoPath)

        #startAt = 350
        # if(cap.get(cv2.CAP_PROP_FRAME_COUNT) < startAt):
        #    startAt = 0

        #cap.set(cv2.CAP_PROP_POS_FRAMES, startAt)

        while True:
            ret_val, frame = cap.read()

            if ret_val == False:
                print("Frame is empty face_recognition_man file")
                time.sleep(1)
                break

            raw_image = frame.copy()
            #print("Frame copied")

            timestamp = datetime.now()
            cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, str(fps), (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            frame_counter += 1

            outputFrame = frame
            # print(fps)
            # time.sleep(0.05)
            # cv2.waitKey(int(900/fps))#240
            # cv2.waitKey(int(999/fps))#240
            #

            time.sleep(0.05)

            #height, width, channels = frame.shape

            #outputFrame = cv2.resize(frame,(int(width/2),int(height/2)))

            if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                cap = cv2.VideoCapture(currentSettings.videoPath)

            if resetVideo:
                cap.release()
                break

    elif currentSettings.currentVideoMode == VideoMode.Live:
        # https://devtalk.nvidia.com/default/topic/1051913/jetson-nano/how-to-close-gstreamer-pipeline-in-python/post/5340291/#5340291

        (live_width, live_height, live_fps) = getResolutionAndFramerate(
            currentSettings.liveResolution)

        print(live_width)
        print(live_height)
        print(live_fps)

        # TODO maybe api
        flipmode = 0

        # TO Restart camera
        # sudo systemctl restart nvargus-daemon

        device_name = "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        device_name += f"width=(int){live_width}, height=(int){live_height},format=(string)NV12, framerate=(fraction){live_fps}/1 "
        device_name += f"! nvvidconv flip-method={flipmode} ! video/x-raw, width=(int){live_width}, height=(int){live_height}, format=(string)BGRx "
        device_name += "! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True "
        #device_name += "! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

        print(device_name)
        time.sleep(2)
        # TODO see to remove the 2 sec sleep

        #cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
        cap = cv2.VideoCapture(device_name)
        if cap.isOpened():
            # cv2.waitKey(10)
            time.sleep(0.1)

            # TODO Get path to save
            # TODO Detect if recording active

            while True:
                # try:
                start = time.process_time()
                ret_val, img = cap.read()

                # print("Cap")

                # if knn != None:
                #    print("KNN READY")

                #denoised = img.copy()

                start_de = time.time()

                #dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #denoised_knn = knn.denoiseFrame(dst)

                #print("denoised: "+ str(time.time() - start_de))

                if ret_val == False:
                    print("Frame is empty detect_motion live")
                    time.sleep(0.5)
                    continue

                raw_image = img.copy()

                # croppedImage = raw_image # CUT X and Y
                # cropp image resize to 2520 * 1080

                if currentSettings.currentDetectionMode == DetectionMode.Record:
                    fr = {}

                    fr["imagecaptured"] = img

                    if q.qsize() < 10:
                        q.put(fr)
                    else:
                        q.put(fr)

                print("new camera frame ready")

                origWidth = 3264  # todo dynamic
                origHeight = 1848  # todo dynamic

                tensorRtWidth = 2520
                tensorRtHeigth = 1080

                strechFactor = origWidth / float(tensorRtWidth)
                neededeHeight = tensorRtHeigth * strechFactor
                obsoleteHeight = origHeight - neededeHeight

                overlay = img.copy()

                #print("origWidth" + str(origWidth))
                #print("strechFactor" + str(strechFactor))
                #print("neededeHeight" + str(neededeHeight))
                #print("obsoleteHeight" + str(obsoleteHeight))

                cv2.rectangle(overlay, (0, 0), (origWidth, int(
                    obsoleteHeight/2)), (0, 0, 255), -1)
                cv2.rectangle(overlay, (0, int(
                    (obsoleteHeight/2) + neededeHeight)), (origWidth, origHeight), (0, 255, 0), -1)

                timestamp = datetime.now()
                cv2.putText(img, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
                            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(img, str(fps), (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                time.sleep(0.01)
                # cv2.waitKey(1)

                outputFrame = img.copy()
                # to see what mtcnn sees

                y_start = int(obsoleteHeight/2)

                #outputFrame = img[y_start:int(y_start+neededeHeight), 0:origWidth]

                cv2.addWeighted(overlay, 0.5, outputFrame, 0.5, 0, outputFrame)

                #print(time.process_time() - start)
                if resetVideo:
                    cap.release()
                    break
                # except:
                #    print("EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE / EXCEPTION CAPTURE")
        else:
            print("camera open failed")
    elif currentSettings.currentVideoMode == VideoMode.Image:
        # reads image 'opencv-logo.png' as grayscale
        frame = cv2.imread(currentSettings.videoPath)
        outputFrame = cv2.resize(frame, (2560, 1440))

        raw_image = frame.copy()

    print("EXIT face_recognition_man")

    #cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")


# TODO Clean up
t = threading.Thread(target=face_recognition_man)


def face_recognition():
    # grab global references to the output frame and lock variables
    #global outputFrame
    global raw_image
    global lastface_img
    global detector
    global encodedImage_face
    global encodedImage_face_recognized
    global faceEvents

    global currentSettings

    #global resnet
    #global device
    #global mtcnn

    # TODO support windows
    #known_image = face_recognition.load_image_file("/home/karlo/faces/noctis.PNG")
    #known_image = face_recognition.load_image_file("I:\\Maturaarbeit\\test\\noctis.PNG")

    #noct_encoding = face_recognition.face_encodings(known_image)[0]

    # https://github.com/jkjung-avt/tensorrt_demos

    # loop over frames from the output stream
    while True:

        # Check if Face detection/recognition is needed currently
        if not (currentSettings.currentDetectionMode == DetectionMode.DetectFaces or
                currentSettings.currentDetectionMode == DetectionMode.RecognizeFaces):
            time.sleep(1)
            continue

        time.sleep(0.01)  # to give the cpu more time
        # wait until the lock is acquired

        frametodetect = None

        #print('wait for frame')
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if raw_image is None:
            time.sleep(0.2)
            continue

        #frametodetect = raw_image.copy()

        #lastface_img = outputFrame

        #encodedImage_face = outputFrame

        # small_img = outputFrame #cv2.resize(outputFrame, (0, 0), fx=0.25, fy=0.25)
        # https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

        start_time = time.time()

        print('DETECTING')

        frametodetect = raw_image
        raw_image = None

        # TODO apply crop settings and ensure them

        # original frame is 3264 x 1848
        #

        origWidth = 3264  # todo dynamic
        origHeight = 1848  # todo dynamic

        tensorRtWidth = 2520
        tensorRtHeigth = 1080

        strechFactor = origWidth / float(tensorRtWidth)
        neededeHeight = int(tensorRtHeigth * strechFactor)
        obsoleteHeight = origHeight - neededeHeight

        y_start = int(obsoleteHeight/2)

        crop_img_frame = frametodetect[y_start:y_start +
                                       neededeHeight, 0:origWidth]

        resizedframetodetect = cv2.resize(crop_img_frame, (2520, 1080))

        results, landmarks = detector.detect(resizedframetodetect, minsize=40)
        print(results)
        print('----------')
        print(landmarks)
        print('{} face(s) found'.format(len(results)))

        print(time.time() - start_time)

        #lastface_img = frametodetect

        rows = 5
        columns = 5

        blank_image = np.zeros((rows * 220, columns * 220, 3), np.uint8)

        currRow = 0
        currCol = 0

        foundFaces = []

        # threading.enumerate()
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(threading.enumerate())
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        for r in results:
            x1, y1, x2, y2, conf = r
            #kp = r['keypoints']
            # TODO take from the original image and consider resize factor to get to the right place for the face
            face = resizedframetodetect[int(y1):int(y2), int(x1):int(x2)]
            #frametodetect = cv2.rectangle(frametodetect, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if face is None:
                continue

            try:
                # Prepare for face recognition
                resizedFace = cv2.resize(face, (220, 220))

                resizedFace160 = cv2.resize(face, (160, 160))

                foundFaces.append(resizedFace160)

                y_offset = currRow * 220
                x_offset = currCol * 220

                if currRow < 5:
                    blank_image[y_offset:y_offset+resizedFace.shape[0],
                                x_offset:x_offset+resizedFace.shape[1]] = resizedFace

                currCol += 1

                if currCol >= 5:
                    currCol = 0
                    currRow += 1
            except:
                print("FAILED TO PROCESS FACE")
                print(frametodetect)
                print("face")
                print(face)

        if currentSettings.currentDetectionMode == DetectionMode.DetectFaces:
            blank_image = cv2.resize(blank_image, (5*80, 5*80))

            (flag, encodedImageRec) = cv2.imencode(".jpg", blank_image)
            encodedImage_face_recognized = encodedImageRec

        if currentSettings.currentDetectionMode == DetectionMode.RecognizeFaces:
            # Recognize faces

            currRow = 0
            currCol = 0

            blank_image_recognized = np.zeros(
                (rows * 160, columns * 160, 3), np.uint8)

            for ff in foundFaces:
                # torch.device(device)

                #tensor_face = F.to_tensor(np.float32(ff))

                # print(tensor_face)

                # TODO process all at once
                aligned = []

                # aligned.append(tensor_face)

                # try:

                foundFaceEmb = FaceNetEncoder.return_emb(ff)

                start_m = time.time()

                #x_aligned, prob = mtcnn(img, return_prob=True)
                # aligned.append(x_aligned)

                # if x_aligned is None:
                #    continue

                #print(time.time() - start_m)

                #tensor_face = F.to_tensor(np.float32(ff))
                # aligned.append(x_aligned)
                #aligned = torch.stack(aligned).to(device)
                #embeddings = resnet(aligned).detach().cpu()

                #currEmb = embeddings[0]

                #print(time.time() - start_m)

                currVal = 999
                currName = "None"

                faceFound = None

                for af in allFaces:
                    print(af.name)

                    calc_emb = FaceNetEncoder.face_distance(
                        af.embedding, foundFaceEmb)
                    print(calc_emb)
                    #(currEmb - af.embedding).norm().item()

                    if calc_emb < currVal:
                        currVal = calc_emb
                        currName = af.name

                        faceFound = af

                    print(calc_emb)

                if currVal > 1.1:
                    currName = "Undef"
                    ff = cv2.putText(ff, str(currName), (10, 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    ff = cv2.putText(ff, str(currName), (10, 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    ff = cv2.putText(ff, str(int(currVal * 100)/float(100)),
                                     (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    foundEntry = False
                    for fe in faceEvents:
                        if fe.faceId == faceFound.faceId:
                            # curr val  Object of type 'float32' is not JSON serializable
                            fe.addEvent(1)
                            foundEntry = True

                    if not foundEntry:
                        # face unknown register it
                        faceEvent = FaceRecognition(
                            faceFound.faceId, faceFound.name)
                        # curr val  Object of type 'float32' is not JSON serializable
                        faceEvent.addEvent(1)
                        faceEvents.append(faceEvent)

                y_offset = currRow * 160
                x_offset = currCol * 160

                txtSize = cv2.getTextSize(
                    str(currName), cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

                #ff = cv2.rectangle(ff, (0, 20), (120, 20), (0,0,0), 10)

                if currRow < 5:
                    blank_image_recognized[y_offset:y_offset +
                                           ff.shape[0], x_offset:x_offset+ff.shape[1]] = ff

                currCol += 1

                if currCol >= 5:
                    currCol = 0
                    currRow += 1

                # except Exception as e:
                #    print(e)
                #    print("EXCEPTION TORCH")

                #[[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]

            blank_image_recognized = cv2.resize(
                blank_image_recognized, (5*80, 5*80))

            (flag, encodedImageRec) = cv2.imencode(
                ".jpg", blank_image_recognized)
            encodedImage_face_recognized = encodedImageRec

        '''
        for r in results:
            x1, y1, width, height = r['box']
            x2, y2 = x1 + width, y1 + height
            kp = r['keypoints']
            face = resizedframetodetect[y1:y2, x1:x2]
            #frametodetect = cv2.rectangle(frametodetect, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if face is None:
                continue
            
            try:
                # Prepare for face recognition
                resizedFace = cv2.resize(face,(220,220))
                y_offset = currRow * 220
                x_offset = currCol * 220

                if currRow < 5:
                    blank_image[y_offset:y_offset+resizedFace.shape[0], x_offset:x_offset+resizedFace.shape[1]] = resizedFace

                currCol += 1

                if currCol >= 5:
                    currCol = 0
                    currRow += 1
            except:
                print("FAILED TO PROCESS FACE")
                print(frametodetect)
                print("face")
                print(face)
            #encodedImage_face = cv2.circle(encodedImage_face,(lex, ley), 5, (0,255,0), -1)

        print(time.time() - start_time)

        '''

        '''
        if len(results) > 0:
            print('FACE DETECTED')
            # extract the bounding box from the first face
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            

            # extract the face
            face = outputFrame[y1:y2, x1:x2]

            # resize pixels to the model size
            #image = Image.fromarray(face)
            lastface_img = face

            # encode the frame in JPEG format
            (flag, encodedImage_face) = cv2.imencode(".jpg", lastface_img)
        else:
            print('NO FACE FOUND')
            continue

        '''

        '''
        print('detection')

        face_locations = face_recognition.face_locations(small_img, model="cnn")
        face_encodings = face_recognition.face_encodings(
                    small_img, 
                    face_locations
                )
        print(face_locations)
        print(face_encodings)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            lastface_img = small_img
            outputFrame = cv2.rectangle(lastface_img, (left, top), (right, bottom), (0, 0, 255), 2)


        if len(face_encodings) > 1:
            results = face_recognition.compare_faces([noct_encoding], face_encodings[0])

            if results[0] == True:
                print("It's a picture of me!")
                print(results[0])
            else:
                print("It's not a picture of me!")

        if lastface_img is None:
            continue

        print("last face is not none")


        (flag, encodedImage) = cv2.imencode(".jpg", lastface_img)
        encodedImage_face = encodedImage

        # ensure the frame was successfully encoded
        if not flag:
            print("flag failed")
            continue
        '''

        # TODO Keep only one -> return one image trough one api endpoint

        #(flag, encodedImage) = cv2.imencode(".jpg", blank_image)
        #encodedImage_face = encodedImage

        # yield the output frame in the byte format
        # yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
        #    bytearray(encodedImage) + b'\r\n')

    print("EXIT face_recognition EXIT face_recognition EXIT face_recognition EXIT face_recognition EXIT face_recognition EXIT face_recognition EXIT face_recognition ")


def yield_return_main_img():
    global outputFrame
    val = 0

    start_time_all = time.time()

    while True:

        start_time = time.time()

        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            #print("OUTPUT FRAME IS NONE")
            time.sleep(0.5)
            continue

        outputFrame2 = cv2.resize(outputFrame, (1280, 720))

        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame2)

        # ensure the frame was successfully encoded
        if not flag:
            continue

        # Terminate after 15 sec since the frontend check for a connection every sec its only needed for that time
        # This is to prevent dead browser sessions using cpu without any need
        if((time.time() - start_time_all) > 500):
            break

        #print("SENDING IMAGES: "+ str(val))

        #print(time.time() - start_time)

        time.sleep(0.2)
        val += 1
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


def yield_return_face_img():
    global encodedImage_face

    start_time_all = time.time()
    while True:
        if encodedImage_face is None:
            time.sleep(0.5)

            blank = np.zeros((5, 5, 3), np.uint8)

            (flag, encBlank) = cv2.imencode(".jpg", blank)

            #print("encodedImage_face_recognized is NONE")
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encBlank) + b'\r\n')

            print("encodedImage_face is NONE")
            break

        # Terminate after 15 sec since the frontend check for a connection every sec its only needed for that time
        # This is to prevent dead browser sessions using cpu without any need
        if((time.time() - start_time_all) > 500):
            break

        time.sleep(0.25)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage_face) + b'\r\n')


def yield_return_face_img_rec():
    global encodedImage_face_recognized

    start_time_face = time.time()

    while True:
        if encodedImage_face_recognized is None:
            time.sleep(0.5)
            blank = np.zeros((5, 5, 3), np.uint8)
            blank[:] = (255, 0, 0)
            (flag, encBlank) = cv2.imencode(".jpg", blank)

            #print("encodedImage_face_recognized is NONE")
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encBlank) + b'\r\n')
            continue

        # Terminate after 15 sec since the frontend check for a connection every sec its only needed for that time
        # This is to prevent dead browser sessions using cpu without any need
        if((time.time() - start_time_face) > 500):
            break

        time.sleep(0.25)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage_face_recognized) + b'\r\n')


@app.route("/api/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)

    return Response(yield_return_main_img(),  # generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/video_feed_faceimage")
def video_feed_faceimage():
    # return the response generated along with the specific media
    # type (mime type)

    # DISABLE FOR NOW
    return ('', 204)

    return Response(yield_return_face_img(),  # last_face(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/video_feed_faceimage_rec")
def video_feed_faceimage_rec():
    # return the response generated along with the specific media
    # type (mime type)

    # DISABLE FOR NOW
    # return ('', 204)

    return Response(yield_return_face_img_rec(),  # last_face(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/last_image")
def video_feed_last_image():
    print("sending last image")
    global outputFrame

    # check if the output frame is available, otherwise skip
    # the iteration of the loop
    if outputFrame is None:
        print("OUTPUT FRAME IS NONE last img")
        return ('', 204)

    # encode the frame in JPEG format
    (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

    return Response(bytearray(encodedImage), mimetype="image/jpeg")


@app.route("/api/detections", methods=['GET'])
def get_detections():
    # print("detections")

    return_data = []

    for fe in faceEvents:
        # print(fe.faceId)
        # print()
        # print(fe.returnGroupByTime(60))
        # print(len(fe.probabilities))

        # TODO this returns wrong value
        isOnline = fe.isOnline(60, 10)

        info = {
            "faceId": fe.faceId,
            "name": fe.name,
            "isOnline": isOnline,
            "groupedHits": fe.returnGroupByTime(60),
            "allRecognizedAt": [],  # This cant be serialized for now -> fe.recognizedAt,
            "allRecognizedAt":  [],  # fe.recognizedAtS, # after a while this object gets big
            "allProbabilities": []  # fe.probabilities # after a while this object gets big
        }

        return_data.append(info)

    # print(return_data)

    return json.dumps(return_data)


@app.route("/api/faces", methods=['GET', 'POST', 'DELETE'])
def faces_info_func():
    global allFaces
    global detector  # To align each saved image
    if request.method == 'GET':

        json_string = json.dumps([face.serialize(True) for face in allFaces])
        # print(json_string)
        return json_string
        """return the information for <user_id>"""

    if request.method == 'POST':

        #

        jsonString = request.get_json()

        newFace = FaceInfo(None)

        newFace.name = jsonString["name"]
        newFace.imageBase64 = jsonString["imageBase64"]

        newFace.loadImage()

        emb = FaceNetEncoder.return_emb(newFace.imageArray)

        newFace.embedding = emb

        newFace.writeToDisk(detector)

        allFaces.append(newFace)

        return "SUCCESS"

    if request.method == 'DELETE':

        faceId = request.args.get('faceId')

        print(faceId)
        main_path = os.path.dirname(os.path.realpath(__file__))

        folder = Path(main_path)
        face_file_name_json = faceId + "_face.json"
        face_file_name_png = faceId + "_face.png"
        file_name_json = folder / "faces" / face_file_name_json
        file_name_png = folder / "faces" / face_file_name_png

        os.remove(file_name_json)
        os.remove(file_name_png)

        for i, faceItem in enumerate(allFaces):
            if faceItem.faceId == faceId:
                del allFaces[i]
                break

        return "DELETED"


@app.route('/api/start', methods=['GET'])
def start():
    global t
    #t = threading.Thread(target=face_recognition_man)
    #t.daemon = False
    # t.start()

    return "started"

#api.add_resource(TodoSimple, '/test')
@app.route('/api/settings', methods=['GET', 'POST'])
def settingsFunc():
    global currentSettings
    global resetVideo
    global t

    print("settings CALL" + request.method)

    if request.method == 'GET':
        return currentSettings.serialize()
        """return the information for <user_id>"""

    if request.method == 'POST':
        """modify/update the information for <user_id>"""
        # you can use <user_id>, which is a str but could
        # changed to be int or whatever you want, along
        # with your lxml knowledge to make the required
        # changes

        # todo better way
        resetVideo = True

        time.sleep(0.5)

        resetVideo = False

        json = request.get_json()

        try:
            if t.is_alive:
                t._stop()
        except:
            print("Couldnt stop thread. Waiting 2 secs more to end it self.")
            time.sleep(2)

        currentSettings.currentDetectionMode = DetectionMode(
            int(json["currentDetectionMode"]))
        currentSettings.currentVideoMode = VideoMode(
            int(json["currentVideoMode"]))
        currentSettings.liveResolution = LiveResolution(
            int(json["liveResolution"]))

        # Read only for now
        #currentSettings.videoIsRunning = False
        currentSettings.videoPath = json["videoPath"]

        print(json)
        print(json["croppSettings"])

        currentSettings.croppSettings.x = int(json["croppSettings"]["x"])
        currentSettings.croppSettings.y = int(json["croppSettings"]["y"])
        currentSettings.croppSettings.height = int(
            json["croppSettings"]["height"])
        currentSettings.croppSettings.width = int(
            json["croppSettings"]["width"])

        # Read only for now
        #currentSettings.croppSettings.originalWidth = int(json["croppSettings.originalWidth"])
        #currentSettings.croppSettings.originalHeight = int(json["croppSettings.originalHeight"])

        print(currentSettings)

        currentSettings.saveToDisk()

        print('reset is currently: ' + str(resetVideo))

        t = threading.Thread(target=face_recognition_man)
        t.daemon = False
        t.start()

        # if currentSettings.currentDetectionMode == DetectionMode.Record:
        #    t_save = threading.Thread(target=process_queue)
        #    t_save.daemon = False
        #    t_save.start()

        return "SUCCESS"


def anounceHostIp():
    # https://raspberrypi.stackexchange.com/questions/83659/python-script-does-not-run-when-called-by-service
    cmd = ['hostname', '-I']
    hostname = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    print(hostname)

    #hostname = "DISABLED"

    to = ['*******@gmail.com']

    subject = 'Jetson Nano Faceview IP'
    body = 'Current hostname is: ' + str(hostname)

    username = "*******@gmail.com"
    password = ""

    message = f"""\
Subject: {subject} - {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

{body}"""

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(username, password)
    server.sendmail(username, to, message)
    server.close()

    print('Hostname send trough email')


if __name__ == '__main__':

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('Running on device: {}'.format(device))

    # start a thread that will perform motion detection
    print(sys.version)
    print(cv2.__version__)

 

    if instantRecord:
        # Ensure these settings
        currentSettings.liveResolution = LiveResolution.Live_3264_1848
        currentSettings.currentVideoMode = VideoMode.Live
        currentSettings.currentDetectionMode = DetectionMode.RecognizeFaces

        currentSettings.saveToDisk()
        t_live_view = threading.Thread(target=live_view)
        t_live_view.daemon = True
        t_live_view.start()

    print('Current OS: ' + os.name)

    files = []

    main_path = os.path.dirname(os.path.realpath(__file__))
    folder = Path(main_path)

    facesFolder = folder / "faces"
    print(facesFolder)

    for r, d, f in os.walk(facesFolder):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))

    for f in files:
        name = os.path.basename(f)

        faceId = name.split("_", 1)[0]
        currentFace = FaceInfo(faceId)

        allFaces.append(currentFace)

    names = []

    if os.name != 'nt':

        print("FACENET EMB 2")
        # just to warm up facenet
        empty = np.zeros((2, 1))
        if runEncoder:

            embReturn = FaceNetEncoder.return_emb(empty)
            print(embReturn)

            print(embReturn.shape)
            print(embReturn.dtype)

        # TODO all at once
        '''
        img = Image.fromarray(face.imageArray)
        x_aligned, prob = mtcnn(img, return_prob=True)

        aligned.append(x_aligned)
        aligned = torch.stack(aligned).to(device)
        embeddings = resnet(aligned).detach().cpu()

        face.embedding = embeddings[0]

        #print(face.embedding)
        '''

        #print("FACENET EMB")
        #emb = FN_Embeddings()
        # emb.return_emb(face.imageArray)

        # TODO verify size for the image input

        #embReturn = FaceNetEncoder.return_emb(face.imageArray)
        #face.embedding = embReturn
        # print(embReturn)

    #dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    #print(pd.DataFrame(dists, columns=names, index=names))

    # test data for recs

    '''
    for fe in allFaces[:2]:   
        print(fe.faceId)
        faceEvent = None
        faceEvent = FaceRecognition(fe.faceId, fe.name)

        for i in range(0, random.randint(100, 250)):
            faceEvent.addEvent(random.uniform(0, 1))

        faceEvents.append(faceEvent)
        print(len(faceEvent.probabilities))
    '''

    # print(faceEvents)

    #img = Image.fromarray(testFaceImgs[0])

    # Get cropped and prewhitened image tensor
    #img_cropped = mtcnn(img)

    # print(img_cropped)

    # Calculate embedding (unsqueeze to add batch dimension)
    #img_embedding = resnet(img_cropped.unsqueeze(0)).detach().cpu()

    # torch.from_numpy(testFaceImgs[0])
    # img_embedding = resnet(torch.from)
    # print(img_embedding)

    #imgs = [torch.as_tensor(np.uint8(img)).float().to(device) for img in testFaceImgs]

    #imgs = torch.stack(imgs).to(device)
    #embeddings = resnet(imgs).detach().cpu()

    #dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    #print(pd.DataFrame(dists, columns=names, index=names))


    print("ONLINE")

    t_face = threading.Thread(target=face_recognition)
    t_face.daemon = True
    t_face.start()

    if os.name != 'nt':
        anounceHostIp()

    app.run(debug=False, host='0.0.0.0')
