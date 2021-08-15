"""
A submodule for handling facial caluculations and rendering
"""

from typing import List
import cv2
import time
import numpy as np
import face_recognition as fr
from PIL import Image

FeaturesConv = {
    "left_eyebrow":"eyebrow",
    "right_eyebrow":"eyebrow",
    "left_eye":"eye",
    "right_eye":"eye",
    "nose_bridge":"nose",
    "nose_tip":"nose",
    "top_lip":"lip",
    "bottom_lip":"lip",
    "chin":"chin",
}

FeatureColours ={
    "eyebrow":[0,0,255],
    "eye"    :[0,127,127],
    "nose"   :[0,255,0],
    "lip"    :[127,127,0],
    "chin"   :[255,0,0]
}

class FaceMapping:
    def __init__(self,locations:dict,downscale:int) -> None:
        self.mappings = {}
        for key,value in locations.items():
            points = [np.array(point) for point in value]
            self.mappings[key] = np.array(points)

        self.downscale = downscale
        self.time = time.time()

    def offset(self,x,y) -> None:
        for key,value in self.mappings.items():
            self.mappings[key] -= np.array([x,y])

def Calculate(image:np.ndarray,*,downscale=2) -> List[FaceMapping]:
    WIDTH,HIEGHT,_ = np.array(image.shape)//downscale
    downscaled = cv2.resize(image,(HIEGHT,WIDTH))

    Faces = fr.face_landmarks(downscaled)
    return [FaceMapping(face,downscale) for face in Faces]

def Render(locations:List[FaceMapping],*,res:tuple=None,image:np.ndarray=None)->np.ndarray:
    if image is None:
        image = np.full((res[1],res[0],3),0,dtype=np.uint8)
    else:
        image = image
    
    for x,face in enumerate(locations):
        Mappings = face.mappings
        for name,points in Mappings.items():
            NewPoints = []
            for point in points:
                NewPoints.append(np.array(point)*face.downscale)
            
            if x == 0:
                clr = FeatureColours[FeaturesConv[name]]
            else:
                clr = (255,255,255)
            if name == 'chin':
                image = cv2.polylines(image,[np.array(NewPoints)],
                    color=clr,
                    isClosed= False,
                    thickness=2
                    )
            else:
                image = cv2.fillPoly(image,[np.array(NewPoints)],
                            color = clr,
                            lineType=cv2.LINE_AA
                            )
    
    return image


def Interpolate(_from:List[FaceMapping],_to:List[FaceMapping])->FaceMapping:
    output = []
    for frm,to in zip(_from,_to):
        interp = {}
        for name in frm.mappings.keys():
            interp[name] = []
            for frmX,toX in zip(frm.mappings[name],to.mappings[name]):
                frmX = frmX * frm.downscale
                toX = toX * to.downscale
                itrp = (frmX+toX)//2
                interp[name].append(itrp)

        output.append(FaceMapping(interp,1))
    return output
