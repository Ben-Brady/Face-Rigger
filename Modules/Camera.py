"""
For handling most of the camera related tasks.
Provided a class for the CaptureDevice
"""
from typing import Union
import cv2
import numpy as np
import multiprocessing as mp


class CaptureDevice():
    DEVICE: Union[str,int]
    FPS: int
    FRAMETIME: int
    ENABLE: bool
    _FrameQueue: mp.Queue
    
    def __init__(self,device:Union[str,int],*,fps:int,rotation:int):
        self.DEVICE = device
        self.FPS = fps
        self.FRAMETIME = 1000//fps
        self.ENABLE = True
        self._FrameQueue = mp.Queue(1)
        
        rotation %= 360
        if rotation == 0: # Convert rotation to cv2.rotation
            self.ROTATION = None
        else:
            self.ROTATION = (rotation // 90) - 1 # 90,180,270 = 0,1,2
        
        
        print('Checking Capture Device')
        cap = cv2.VideoCapture(device)
        ret,image = cap.read()
        
        if ret == False:
            print('Capture Device Error')
            exit()
        
        print('Capture Device OK')
        cap.release()

        if self.ROTATION:
            cv2.rotate(image,self.ROTATION)
        
        self.RESOLUTION = image.shape[:2]
        self.WIDTH = self.RESOLUTION[1]
        self.HEIGHT = self.RESOLUTION[0]
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

                try:
                    self._FrameQueue.put_nowait(image)
                except Exception:
                    pass
        finally:
            cap.release()
            self.ENABLE = False

    @property
    def image(self) -> np.ndarray:
        return self._FrameQueue.get()