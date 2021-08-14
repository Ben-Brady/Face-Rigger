from Modules import FaceRender,Camera
import os
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue


DEVICE = Camera.CaptureDevice(0,fps=60,rotation=cv2.ROTATE_90_COUNTERCLOCKWISE)
DOWNSCALE = 2
FRAMEBUFFER = 1

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

print("\033[s")
FrameTypes = [' ' for x in range(DEVICE.FPS)]
FrameTimes = [0 for x in range(DEVICE.FPS)]
while DEVICE.ENABLE and cv2.waitKey(1000//DEVICE.FPS):
    start = time.time()
    print("\033[u") # Moves the cursor to the top left

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
    Image = FaceRender.Render(face,image=DEVICE.image)

    cv2.imshow('Render',Image)

    # Interpolation Ration
    FrameTypes.insert(0,type)
    FrameTypes.pop(len(FrameTypes)-1)
    print(f"Face Data:{''.join(FrameTypes)}")
    if FrameTypes.count('I'):
        InterpRation = FrameTypes.count('P') / FrameTypes.count('I')
    else:
        InterpRation = 1
    print(f"Interpolation Ratio: {InterpRation:.2f}")
    
    # Frame Timing
    frametime = int((time.time() - start)*1000)
    FrameTimes.pop(0)
    FrameTimes.append(frametime)
    print(f"Frame Time: {frametime}ms***")
    print(f"Average Frame Time: {np.mean(FrameTimes):.2f}ms***")