from enum import IntEnum     # for enum34, or the stdlib version

__all__ = ['VideoMode', 'DetectionMode', 'LiveResolution']


class VideoMode(IntEnum):
    File = 0,
    Live = 1,
    Image = 2

class DetectionMode(IntEnum):
    Nothing = 0,
    DetectFaces = 1,
    RecognizeFaces = 2,
    Record = 3

class LiveResolution(IntEnum):
    Live_3264_2464 = 0,
    Live_3264_1848 = 1,
    Live_1920_1080 = 2,
    Live_1280_720 = 3

    
#GST_ARGUS: Available Sensor modes :
#GST_ARGUS: 3264 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
#GST_ARGUS: 3264 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
#ST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
#GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
#GST_ARGUS: 1280 x 720 FR = 120.000005 fps Duration = 8333333 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
