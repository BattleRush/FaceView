from .enums import *
import json
import os
import __main__

class CroppSettings(object):
    x = 0
    y = 0
    height = 0
    width = 0

    originalHeight = 0
    originalWidth = 0

    def __init__(self):
        self.x = 0
        self.y = 0
        self.height = 0
        self.width = 0
    
    def serialize(self):
        return {
            "x": self.x,
            "y": self.y,
            "height": self.height,
            "width": self.width,
            "originalHeight": self.originalHeight,
            "originalWidth": self.originalWidth
        }


class SettingsInfo(object):    
    currentDetectionMode = DetectionMode.Nothing
    currentVideoMode = VideoMode.File
    liveResolution = LiveResolution.Live_1280_720
    videoIsRunning = False
    videoPath = "I:\\Maturaarbeit\\vid.mp4"

    # INIT Empty for now
    croppSettings = None

    def __init__(self, detMode, vidMode, path):
        self.currentDetectionMode = detMode
        self.currentVideoMode = vidMode
        self.videoIsRunning = False
        self.videoPath = path
        self.croppSettings = CroppSettings()
    
    def serialize(self):
        return {
            "currentDetectionMode": self.currentDetectionMode.value,
            "currentVideoMode": self.currentVideoMode.value,
            "liveResolution": self.liveResolution.value,
            "videoIsRunning": self.videoIsRunning,
            "videoPath": self.videoPath,
            "croppSettings": self.croppSettings.serialize()
        }

    def saveToDisk(self):
        main_path = os.path.dirname(__main__.__file__)
        file_name = main_path + "/settings.json" # Wont work for NT
        print(file_name)
        settings_file = open(file_name, "w")
        settings_file.write(json.dumps(self.serialize(), indent=4))
        settings_file.close()

    def readFromDisk(self):
        main_path = os.path.dirname(__main__.__file__)
        file_name = main_path + "/settings.json" # Wont work for NT
        if not os.path.isfile(file_name):
            return

        settings_file = open(file_name, "r")
        settings_info = settings_file.read()
        try:
            settings_json = json.loads(settings_info)

            self.currentDetectionMode = DetectionMode(int(settings_json["currentDetectionMode"]))
            self.currentVideoMode = VideoMode(int(settings_json["currentVideoMode"]))
            self.liveResolution = LiveResolution(int(settings_json["liveResolution"]))
            # Read only for now
            #currentSettings.videoIsRunning = False
            self.videoPath = settings_json["videoPath"]

            self.croppSettings.x = int(settings_json["croppSettings"]["x"])
            self.croppSettings.y = int(settings_json["croppSettings"]["y"])
            self.croppSettings.height = int(settings_json["croppSettings"]["height"])
            self.croppSettings.width = int(settings_json["croppSettings"]["width"])
        except:
            print("Failed to load settings.json")

