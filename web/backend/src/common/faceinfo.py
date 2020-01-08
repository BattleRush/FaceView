import uuid
import cv2
from pathlib import Path
import base64
from io import StringIO, BytesIO
from PIL import Image
import numpy as np
import os
import json
import __main__

main_path = os.path.dirname(__main__.__file__)

class FaceInfo(object):
    faceId = None
    name = ""
    imageBase64 = None 
    imageArray = None

    lastSeen = None

    embedding = None

    # TODO store in face folder

    def __init__(self, faceId):
        if faceId is None:
            return

        self.faceId = faceId

        folder = Path(main_path)
        face_file_name = self.faceId + "_face.json"
        file_name = folder / "faces" / face_file_name


        if not os.path.isfile(str(file_name)):
            return

        face_file = open(file_name, "r")
        face_info = face_file.read()

        try: 
            face_json = json.loads(face_info)
            self.name = face_json["name"]
            self.imageBase64 = face_json["imageBase64"]
            embeddingString = face_json["embedding"]


            memfile = BytesIO()
            memfile.write(json.loads(embeddingString).encode('latin-1'))
            memfile.seek(0)
            a = np.load(memfile)

            self.embedding = a #np.fromstring(embeddingString, dtype="float32")#.reshape(512,) maybe not needed

            self.loadImage()
        except:
            print("Failed to read: " + str(file_name))



    def loadImage(self):
        clean_base64 = self.imageBase64.split(",",1)[1] 

        # TODO Create method
        buf = BytesIO()
        buf.write(base64.b64decode(clean_base64))
        imageInfo = Image.open(buf)

        # TODO verify if bgr is right
        self.imageArray = cv2.cvtColor(np.array(imageInfo), cv2.COLOR_RGB2BGR)

        # TODO do this before
        self.imageArray = cv2.resize(self.imageArray, (160, 160))

        #print(self.imageArray)

    def serialize(self, isWeb):
        a = self.embedding
        memfile = BytesIO()
        np.save(memfile, a)
        memfile.seek(0)
        serialized = json.dumps(memfile.read().decode('latin-1'))

        if self.embedding is None:
            self.embedding = []

        if isWeb:
            return {
                "faceId": self.faceId,
                "name": self.name,
                "imageBase64": self.imageBase64,
                "embedding": self.embedding.tolist()
                #lastSeen
            }

        else:
            
            return {
                "faceId": self.faceId,
                "name": self.name,
                "imageBase64": self.imageBase64,
                "embedding": serialized
                #lastSeen
            }

        
    def writeToDisk(self, detector):
        
        folder = Path(main_path)

        if not os.path.isdir(folder / "faces"):
            os.mkdir(folder / "faces")


        self.faceId = uuid.uuid4().hex
        face_file_name = self.faceId + "_face.json"
        file_name = folder / "faces" / face_file_name

        # Make sure the image is 100 x 100
        self.convertBase64AndResize(detector)

        face_file = open(file_name, "w")
        face_file.write(json.dumps(self.serialize(False), indent=4))
        face_file.close()

        # TODO write face img next to the json file
        # TODO maybe use image already as png

    # TODO rewrite
    def convertBase64AndResize(self, detector):
        #print("convertBase64AndResize")
        
        #print(self.imageBase64)

        clean_base64 = self.imageBase64.split(",",1)[1] 

        #print(clean_base64)

        buf = BytesIO()
        buf.write(base64.b64decode(clean_base64))
        imageInfo = Image.open(buf)

        image = cv2.cvtColor(np.array(imageInfo), cv2.COLOR_RGB2BGR)
        resizedImg = cv2.resize(image, (220, 220), interpolation=cv2.INTER_AREA)

        # TODO Duplicate

        
        folder = Path(main_path)
        face_file_name = self.faceId + "_face.png"
        file_name = folder / "faces" / face_file_name

        #print(file_name)

        cv2.imwrite(str(file_name), resizedImg) 

        #print(resizedImg)
        #print("@@@@@@@@@@@@@@2")
        #print(self.imageBase64)
        #print("--------------")
        self.imageBase64 =  "data:image/png;base64," + base64.b64encode(cv2.imencode('.png', resizedImg)[1]).decode()
        #print(self.imageBase64)

        imgHeight, imgWidth, d = image.shape

        if imgHeight > 1080:
            # Image need resizing
            imgWidth = int(round((1080/float(imgHeight))*imgWidth))
            imgHeight = 1080

            image = cv2.resize(image, (imgWidth, imgHeight), interpolation=cv2.INTER_AREA)


        blank_image = np.zeros((1080,2520,3), np.uint8)
        blank_image[0:imgHeight, 0:imgWidth] = image
        results, landmarks = detector.detect(blank_image, minsize=40)

        print(results)
        print(landmarks)

        #TODO only once

        for r in results:
            x1, y1, x2, y2, conf = r
            #kp = r['keypoints']
            face = blank_image[int(y1):int(y2), int(x1):int(x2)]
            
            face_file_name = self.faceId + "_face_align.png"
            file_name = folder / "faces" / face_file_name

            cv2.imwrite(str(file_name), face) 

            
            face_file_name = self.faceId + "_face_blank.png"
            file_name = folder / "faces" / face_file_name

            cv2.imwrite(str(file_name), blank_image) 
