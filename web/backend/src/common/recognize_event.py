
import time
from datetime import datetime, date, timedelta
import random
from collections import OrderedDict

# TODO add this to the face object
class FaceRecognition(object):
    #faceId = ""

    #recognizedAt = []
    #probabilities = []

    # TODO rename ms since midnight now tracking unix time
    
    def __init__(self, faceId, name):
        self.faceId = faceId
        self.name = name

        # TODO make one variable
        self.recognizedAt = []
        self.recognizedAtS = []
        self.probabilities = []

    # rename probability with euclid distance or something
    def addEvent(self, probability):
        # add seconds only for testing
        repeat = random.randint(1, 600)

        # TODO add these from the detection method
        now = datetime.now()# - timedelta(seconds=repeat)
        #ms_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000
        #ms_since_midnight = now.timestamp() * 1000

        #ms_since_midnight -= repeat * 1000

        self.recognizedAt.append(now)
        self.recognizedAtS.append(now.timestamp() * 100)
        self.probabilities.append(probability)

    def returnGroupByTime(self, seconds):  
        # TODO check if better to pass probabilities
        # To prevent writes while reading 
        # TODO verify perfomance and if needed

        recAtCopy = self.recognizedAt
        probabilitiesCopy = self.probabilities

        grouped = []

        grouped_data = {}

        #default_data['item3'] = 3
        
        for i, rec in enumerate(recAtCopy):
            #https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
            #tm - datetime.timedelta(minutes=tm.minute % 10,
            #                 seconds=tm.second,
            #                 microseconds=tm.microsecond)


            #roundedBySecs = int(rec / seconds) * seconds * 1000

            #print(rec)

            #hours=rec.hour,
            #minutes=rec.minute,
            tm = rec - timedelta(
                            seconds=rec.second % seconds)

            roundedBySecs = int(tm.timestamp())
            #print(tm)
            #print(roundedBySecs)     

            #'a' in x

            if roundedBySecs in grouped_data:
                grouped_data[roundedBySecs] += 1
            else:
                grouped_data[roundedBySecs] = 1

        D1 = dict(OrderedDict(sorted(grouped_data.items(), key = lambda t: t[0])))


        for x in D1:
            grouped.append({"time":x,
            "value": D1[x]})


        return grouped

        #for idx, val in enumerate(ints):
        #    print(idx, val)

    def isOnline(self, seconds, minHits):
        # TODO add these from the detection method
        # TODO create method
        now = datetime.now()
        #ms_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000
        ms_since_midnight = now.timestamp() * 1000

        ms_since_midnight -= seconds * 1000

        if ms_since_midnight < 0:
            ms_since_midnight = 0

        # To prevent writes while reading 
        # TODO verify perfomance and if needed
        recAtCopy = self.recognizedAt
        probabilitiesCopy = self.probabilities

        currIndex = len(self.recognizedAt)

        currMinHits = minHits



        for recAt in recAtCopy[::-1]:
            if recAt.timestamp() * 1000 > ms_since_midnight:
                break
            currMinHits -= 1

            if currMinHits < 1:
                return True

        return False
            





