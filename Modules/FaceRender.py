"""
A submodule for handling facial caluculations and rendering
"""

from typing import List
import cv2
import colorsys
import time
import numpy as np
import face_recognition as fr
from PIL import ImageColor

FeaturesConv = {
    "left_eyebrow"  : "eyebrow",
    "right_eyebrow" : "eyebrow",
    "left_eye"      : "eye",
    "right_eye"     : "eye",
    "nose_bridge"   : "nose",
    "nose_tip"      : "nose",
    "top_lip"       : "lip",
    "bottom_lip"    : "lip",
    "chin"          : "chin",
}

FeatureColours ={
    "eyebrow" : "#7518B8",
    "eye"     : "#B53E6F",
    "nose"    : "#1FDB8D",
    "lip"     : "#884FB0",
    "chin"    : "#C2115B",
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
        image = np.zeros((res[0],res[1],3),dtype=np.uint8)
    else:
        image = image

    AllPoints = []
    for x,face in enumerate(locations):
        Mappings = face.mappings
        for name,points in Mappings.items():
            Feature = FeaturesConv[name]
            NewPoints = []
            for point in points:
                NewPoints.append(np.array(point)*face.downscale)
                
            AllPoints.extend(NewPoints)
            clr = FeatureColours[Feature]
            if isinstance(clr,str): # If a hex string is provided
                clr = clr[-6:] # Normalise without the #
                clr = ImageColor.getrgb("#"+clr) # Convert to RGB

            if Feature == "chin":
                image = cv2.polylines(image,[np.array(NewPoints)],
                            color=clr,
                            isClosed=False,
                            thickness=4,
                            lineType=cv2.LINE_AA
                        )
                
            elif Feature == 'eye':
                image = cv2.polylines(image,[np.array(NewPoints)],
                            color=clr,
                            isClosed= True,
                            thickness=1,
                            lineType=cv2.LINE_AA
                        )
            else:
                image = cv2.fillPoly(image,[np.array(NewPoints)],
                            color = clr,
                            lineType=cv2.LINE_AA
                        )
                
    for point in AllPoints:
        image = cv2.circle(image,tuple(point),2,(255,255,255),-1)
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
