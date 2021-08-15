from Modules import FaceRender,Camera
import os
import argparse
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue

ArgParser = argparse.ArgumentParser()

ArgParser.add_argument("-c","--camera",   type=int,  default=0, help="Device number of the camera to use")
ArgParser.add_argument("-f","--fps",      type=int,  default=30,help="Frames per second of the video")
ArgParser.add_argument("-r","--rotation", type=int,  default=0, help="Rotation of the camera in 90° increments. e.g. 1=90°, 2=180°")
ArgParser.add_argument("-d","--downscale",type=int,  default=2, help="How much to downscale the video before calculations, lower is faster but less accurate")
ArgParser.add_argument("-b","--buffer",   type=int,  default=2, help="Buffer size for facial detection, helps to reduce jitter but increases latency")
args =  vars(ArgParser.parse_args())

print(args)

DEVICE = Camera.CaptureDevice(args['camera'],
    fps=args['fps'],
    rotation=args["rotation"]
    )
DOWNSCALE = args["downscale"]
FRAMEBUFFER = args["buffer"]

def CaptureFaceLandmarks(queue:Queue):
    while DEVICE.ENABLE and cv2.waitKey(1000//DEVICE.FPS):
        image = DEVICE.image
        FaceCalculation = FaceRender.Calculate(image=image,downscale=DOWNSCALE)
        if FaceCalculation:
            if queue.full():
                queue.get()
            queue.put(FaceCalculation)


q = Queue(FRAMEBUFFER)
p = Process(target=CaptureFaceLandmarks, args=(q,))
p.start()
print("Detecting Face...")
Frames = [q.get(),q.get()]

frameTypes = [' ' for x in range(DEVICE.FPS)]
frameTimes = [ 0  for x in range(DEVICE.FPS)]
print("\033[s") # Saves cursor position
while DEVICE.ENABLE and cv2.waitKey(1000//DEVICE.FPS):
    start = time.time()
    print("\033[u") # Loads cursor position
    # Face Generation
    if q.empty():
        type = 'I'
        face = FaceRender.Interpolate(*Frames)
        Frames[0] = face
    else:
        type = 'P'
        face = q.get()
        Frames.pop(0)
        Frames.append(face)

    # Frame Handled
    Image = FaceRender.Render(face,
                            #   image=DEVICE.image
                              res=DEVICE.RESOLUTION
                              )

    cv2.imshow('Render',Image)

    # Interpolation Ration
    frameTypes.insert(0,type) # Insert type in front
    frameTypes.pop(len(frameTypes) - 1) # Remove type from end
    InterpRation = frameTypes.count('I') / DEVICE.FPS
    
    
    # Frame Timing
    frametime = int((time.time() - start)*1000)
    frameTimes.pop(0)
    frameTimes.append(frametime)
    avgFrameTime = int(np.mean(frameTimes))
    
    print(f"{'Data Source:':>30}    {''.join(frameTypes)}") # Prints frametype arrays
    print(f"{'Interpolation Ratio:':>30}    {InterpRation:.2f}")
    print(f"{'Frame Time:':>30}{frametime:>6}ms")
    print(f"{'Average Frame Time:':>30}{avgFrameTime:>6}ms")