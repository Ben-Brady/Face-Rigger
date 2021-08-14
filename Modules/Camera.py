from typing import Union
import cv2
import numpy as np
from multiprocessing import Process,Queue



class CaptureDevice():
    def __init__(self,device:Union[str,int],*,fps=30,downscale=None,rotation=None):
        print('Checking Capture Device')
        cap = cv2.VideoCapture(device)
        ret,image = cap.read()
        if ret:
            print('Capture Device OK')
            cap.release()
        else:
            print('Capture Device Error')
            exit()
        
        self.DEVICE = device
        self.FPS = fps
        self.ROTATION = rotation
        print(self.ROTATION)
        self._FrameQueue = Queue(1)
        self.ENABLE = True
        self.RESOLUTION = image.shape[:2]
        if downscale:
            self.RESOLUTION = self.RESOLUTION[0]//downscale,self.RESOLUTION[1]//downscale
        
        Process(target=self._FrameCapture).start()
        
    
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

                # Resize if the frame needs to be resized
                if image.shape[:2] != self.RESOLUTION: 
                    image = cv2.resize(image,self.RESOLUTION)
                
                try:
                    self._FrameQueue.put_nowait(image)
                except:
                    pass
        finally:
            cap.release()

    @property
    def image(self) -> np.ndarray:
        return self._FrameQueue.get()