import cv2 as cv
import numpy as np

def bodyToImg(r,angle):
    """Function which superimposes a clear marker over Starfield's body radial selector.
    It allows users to more easily select the appropriate settings as the parameters, i.e. radius and angle, are difficult to correctly translate

    Args:
        r (_type_): _description_
        angle (_type_): _description_
    """
    # Load radial selector
    image = cv.imread("assets/RadialSelector.png")
    # y, x
    # center 359x/322y
    # horizontally, end is at roughly at 547
    # Therefore, endpoint of radius is at (547,322)
    # the radius length in pixels is the difference 547-359 which can then be scaled
    CENTERX = 359
    CENTERY = 322
    HORIZONTAL_END_PIXEL = 547
    radiusPixels = HORIZONTAL_END_PIXEL-CENTERX
    angleRadians = angle*(np.pi/180)
    pointVector= np.array([CENTERX + (radiusPixels*r)*np.cos(angleRadians),
                           CENTERY + (radiusPixels*r)*np.sin(angleRadians)]).astype(np.int32)
    
    image = cv.circle(image, pointVector, radius=11, color=(0, 0, 0), thickness=2)
    image = cv.circle(image, pointVector, radius=9, color=(0, 0, 255), thickness=-1)
    return image    

def headshapeToImg(direction,intensity):
    image = cv.imread("assets/CrossSelector.png")
    # center 286x, 234y
    # horizontally end is at 446
    # The radius in pixels is therefore the difference between the centre and endpoint
    CENTERX = 286
    CENTERY = 234
    HORIZONTAL_END_PIXEL = 446
    radiusPixels = HORIZONTAL_END_PIXEL-CENTERX
    
    # The cross shaped selector is analagous to the circle from bodyToImg, however with discrete angles
    r = intensity if intensity != None else 0
    angle = 0
    if direction == "left":
        angle = 180
    elif direction == "right":
        angle = 0
    elif direction == "up":
        angle = -90
    elif direction == "down":
        angle = 90
    
    angleRadians = angle*(np.pi/180)
    pointVector= np.array([CENTERX + (radiusPixels*r)*np.cos(angleRadians),
                           CENTERY + (radiusPixels*r)*np.sin(angleRadians)]).astype(np.int32)
    image = cv.circle(image, pointVector, radius=11, color=(0, 0, 0), thickness=2)
    image = cv.circle(image, pointVector, radius=9, color=(0, 0, 255), thickness=-1)
    return image
    
if __name__ == "__main__":
    bodyToImg(0.75,234)
    headshapeToImg("down",0.24)