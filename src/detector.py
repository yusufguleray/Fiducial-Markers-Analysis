import cv2
import numpy as np
from apriltag import apriltag

def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

#for testing
def drawPoly(img, imgpts):
    cv2.polylines(img,np.int32([imgpts]),True,(255,0,0))
    return img

def drawCordinate(img, corners, imgpts):

    corner = tuple(corners[0].astype(int).ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

    corner = tuple(np.rint([corners]))
    print('Corner', corner)
    print('Impts', imgpts)
    img = cv2.line(img, tuple(np.rint([corner])), tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

imagepath = 'singleTag_Color.png'
image = cv2.imread(imagepath)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
detector = apriltag("tagStandard41h12")
detections = detector.detect(gray)

# Load previously saved data
with np.load('realsense_d415_010721_2.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

print(mtx)

a = [0,0,0]
b = [1,0,0]
c = [0,1,0]
d = [1,1,0]


objp = np.array([a, c, d, b],np.float32) #very promissing
# objp = np.array([b, a, c, d],np.float32) very promissing
# objp = np.array([d, b, a, c],np.float32) very promissing




# axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
#                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

#axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
#                   [0,0,1],[0,1,1],[1,1,1],[1,0,-1] ])

axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

# For drawing the coordinate system
# axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)


for tag in detections:
    center = tag["center"]
    corners2 = tag["lb-rb-rt-lt"]

    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    image = drawCube(image, imgpts)
    # image = drawCordinate(image, corners2, imgpts)
    # corners2 = corners2.reshape((-1,1,2))
    # image = drawPoly(image, corners2)

cv2.imshow('Cubes around the tags',image)
k = cv2.waitKey(0) & 0xFF
if k == ord('s'):
    cv2.imwrite('result.png', image)

