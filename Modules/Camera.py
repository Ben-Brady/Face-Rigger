"""
For handling most of the camera related tasks.
Provided a class for the CaptureDevice
"""
from typing import Union
import cv2
import numpy as np
import multiprocessing as mp


class CaptureDevice():
    def __init__(self,device:Union[str,int],*,fps:int,rotation:int,show:bool):
        self.DEVICE = device
        self.FPS = fps
        self.FRAMETIME = 1000//fps
        self.SHOW = show
        self.ENABLE = True
        rotation = (rotation % 360)
        self._FrameQueue = mp.Queue(1)
        if rotation: # Convert rotation to cv2.rotation
            self.ROTATION = (rotation // 90) - 1 # 90,180,270 = 0,1,2
        else:
            self.ROTATION = None
        
        print('Checking Capture Device')
        cap = cv2.VideoCapture(device)
        ret,image = cap.read()
        
        if ret:
            print('Capture Device OK')
            cap.release()
        else:
            print('Capture Device Error')
            exit()

        if self.ROTATION:
            cv2.rotate(image,self.ROTATION)
        
        self.RESOLUTION = image.shape[:2]
        self.WIDTH = self.RESOLUTION[0]
        self.HEIGHT = self.RESOLUTION[1]
        mp.Process(target=self._FrameCapture).start()
        
    
    def _FrameCapture(self) -> None:
        cap = cv2.VideoCapture(self.DEVICE)
        try:
            while cv2.waitKey(1000//self.FPS):
                ret,image = cap.read()
                if not ret:
                    cap.release()
                    self.ENABLE = False
                    raise LookupError('Frame Capture Error')
                
                # Rotation if the frame needs to be rotated
                if self.ROTATION != None:
                    image = cv2.rotate(image,self.ROTATION)

                # TODO: Prevent skipped frames
                try:
                    self._FrameQueue.put_nowait(image)
                except Exception:
                    pass
                
                if self.SHOW:
                    cv2.imshow('Camera',image)
        finally:
            cap.release()
            self.ENABLE = False

    @property
    def image(self) -> np.ndarray:
        return self._FrameQueue.get()