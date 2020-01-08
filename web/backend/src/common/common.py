from .enums import LiveResolution


def getResolutionAndFramerate(liveResolution):
    live_width = 1280
    live_height = 720
    live_fps = 24

    if liveResolution == LiveResolution.Live_3264_2464:
        live_width = 3264
        live_height = 2464
        live_fps = 5
    elif liveResolution == LiveResolution.Live_3264_1848:
        live_width = 3264
        live_height = 1848
        live_fps = 5 # Fallback to 5 or so maybe
    elif liveResolution == LiveResolution.Live_1920_1080:
        live_width = 1920
        live_height = 1080
        live_fps = 10
    elif liveResolution == LiveResolution.Live_1280_720:
        live_width = 1280
        live_height = 720
        live_fps = 30


    return (live_width, live_height, live_fps)