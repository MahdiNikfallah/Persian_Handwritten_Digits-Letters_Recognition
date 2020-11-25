import numpy as np
import cv2

dataset_folder = ".\\phase1_dataset\\"

Image = cv2.imread(dataset_folder + '7.jpg')

Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
Corners, IDs, _ = cv2.aruco.detectMarkers(Image, Dictionary)
Corners = np.array(Corners)

Markers = {'TopLeft': np.reshape(Corners[IDs == 34], (4, 2)), 'TopRight': np.reshape(Corners[IDs == 35], 
                                                                                     (4, 2)),
           'BottomLeft': np.reshape(Corners[IDs == 33], (4, 2)), 'BottomRight': np.reshape(Corners[IDs == 36],
                                                                                           (4, 2))}

TopLeftCorner = tuple(Markers['TopLeft'][3])
TopRightCorner = tuple(Markers['TopRight'][2])
BottomLeftCorner = tuple(Markers['BottomLeft'][0])
BottomRightCorner = tuple(Markers['BottomRight'][1])
SheetCorners = np.array([TopLeftCorner, TopRightCorner,
                         BottomLeftCorner, BottomRightCorner], dtype=np.float32)

ImageCorners = np.array([(0, 0), (719, 0),
                         (0, 719), (719, 719)], dtype=np.float32)

PerspectiveTransformMatrix = cv2.getPerspectiveTransform(SheetCorners, ImageCorners)
SheetImage = cv2.warpPerspective(Image, PerspectiveTransformMatrix,  (720, 720))

# Calculate Boundings Of Markers To Remove Them
TempPoint = np.append(Markers['TopLeft'][2], [1])
MarkersBoundings = np.matmul(PerspectiveTransformMatrix, TempPoint)
MarkersBoundings = MarkersBoundings / MarkersBoundings[2]
MarkersBoundingsX = int(MarkersBoundings[1])
MarkersBoundingsY = int(MarkersBoundings[0])

GraySheetImage = cv2.cvtColor(SheetImage, cv2.COLOR_BGR2GRAY)
BlurredSheetImage = cv2.GaussianBlur(GraySheetImage, (7, 7), 0)

EdgesSheetImage = cv2.Canny(BlurredSheetImage, threshold1=50, threshold2=120)


cv2.imshow("test",SheetImage)
cv2.waitKey()
cv2.destroyAllWindows()


# Remove Markers for Detecting Boxes
Threshold = 20
SheetImageWithoutMarkers = np.copy(EdgesSheetImage)
SheetImageWithoutMarkers[0: MarkersBoundingsX + Threshold,
                         0: MarkersBoundingsY + Threshold] = 0
SheetImageWithoutMarkers[SheetImageWithoutMarkers.shape[0] - MarkersBoundingsX - Threshold:,
                         0: MarkersBoundingsY + Threshold] = 0
SheetImageWithoutMarkers[0: MarkersBoundingsX + Threshold,
                         SheetImageWithoutMarkers.shape[1] - MarkersBoundingsY - Threshold:] = 0
SheetImageWithoutMarkers[SheetImageWithoutMarkers.shape[0] - MarkersBoundingsX - Threshold:,
                         SheetImageWithoutMarkers.shape[1] - MarkersBoundingsY - Threshold:] = 0


th2 = cv2.adaptiveThreshold(BlurredSheetImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,19,2.2)
    
th2 = cv2.bitwise_not(th2)
kernelC = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
th2 = cv2.morphologyEx(th2,cv2.MORPH_CLOSE, kernelC)

cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\testc2' + '.jpg',th2)

img = cv2.imread(("testc2.jpg"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()

# Structuring Element
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# Create an empty output image to hold values
thin = np.zeros(img.shape,dtype='uint8')
cv2.imshow('thinned',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Loop until erosion leads to an empty set
while (cv2.countNonZero(img1)!=0):
    # Erosion
    erode = cv2.erode(img1,kernel)
    # Opening on eroded image
    opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
    # Subtract these two
    subset = erode - opening
    # Union of all previous sets
    thin = cv2.bitwise_or(subset,thin)
    # Set the eroded image for next iteration
    img1 = erode.copy()

thin = cv2.dilate(thin,kernel)
cv2.imshow("test",thin)
cv2.waitKey()
cv2.destroyAllWindows()


BlockSize = 3
SobelKernelSize = 3
Alpha = 0.01

HarrisScores = cv2.cornerHarris(thin, BlockSize, SobelKernelSize, Alpha)
HarrisScores = HarrisScores / HarrisScores.max()
ThresholdedHarrisScores = np.uint8(HarrisScores > 0.2) * 255

nC, CC, stats, centroids = cv2.connectedComponentsWithStats(ThresholdedHarrisScores)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(ThresholdedHarrisScores, np.float32(centroids), (5, 5), (-1, -1), criteria)


tempp = np.copy(SheetImage)
for j in range(len(corners)):
    cv2.circle(tempp, (int(corners[j][0]), int(corners[j][1])), 1, (255, 255, 255))

cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\Thresh' + '.jpg', ThresholdedHarrisScores)

Cn = [
        np.array([38, 188], np.float),  # Student Id Corner
        np.array([37, 271], np.float),  # First Name Corner
        np.array([37, 356], np.float),  # Last Name Corner
        np.array([68, 457], np.float),  # Is PHD Corner
        np.array([205, 456], np.float), # Is MS Corner
        np.array([404, 455], np.float)  # Is BC Corner
    ]


detected_corners = [None, None, None, None, None, None]
for i in range(0, len(Cn)):
    tmp_corner = Cn[i]
    cv2.circle(SheetImage, (int(tmp_corner[0]), int(tmp_corner[1])), 10, (255, 255, 255))
    dist = 999999
    for j in range(len(corners)):
        tempDist = np.linalg.norm(corners[j] - tmp_corner)
        if tempDist < dist:
            dist = tempDist
            debugIndex = j
            detected_corners[i] = corners[j]
            last_start = int(tmp_corner[0]), int(tmp_corner[1])
            
for j in range(len(detected_corners)):
    cv2.circle(SheetImage, (int(detected_corners[j][0]), int(detected_corners[j][1])), 1, (255, 255, 255))
    
cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\Sheet' + '.jpg', SheetImage)
    
cv2.imshow("test",SheetImage)
cv2.waitKey()
cv2.destroyAllWindows()

TopLeftCornerOfBoxX = int(detected_corners[0][0]) + 6
TopLeftCornerOfBoxY = int(detected_corners[0][1])
temp1 = TopLeftCornerOfBoxX
temp2 = TopLeftCornerOfBoxY
xThreshold = 3
yThreshold = 5
xOffset = 8;
RectangleBoxHeight = 60
RectangleBoxWidth = 478
for i in range(1, 9):
    CroppedImage = SheetImage[TopLeftCornerOfBoxY + xThreshold: TopLeftCornerOfBoxY + RectangleBoxHeight,
                              TopLeftCornerOfBoxX + yThreshold + ((i - 1) * RectangleBoxWidth // 8):
                              TopLeftCornerOfBoxX + (i * (RectangleBoxWidth // 8)) - xThreshold]
    cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\ID'
                + str(i) + '.jpg', CroppedImage)

TopLeftCornerOfBoxX = int(detected_corners[1][0]) + xOffset
TopLeftCornerOfBoxY = int(detected_corners[1][1])

for i in range(1, 9):
    CroppedImage = SheetImage[TopLeftCornerOfBoxY + xThreshold: TopLeftCornerOfBoxY + RectangleBoxHeight,
                              TopLeftCornerOfBoxX + yThreshold + ((i - 1) * RectangleBoxWidth // 8):
                              TopLeftCornerOfBoxX + (i * (RectangleBoxWidth // 8))- xThreshold]
    
    if np.var(CroppedImage) > 175: 
        cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\FN' 
                + str(i) + '.jpg', CroppedImage)

TopLeftCornerOfBoxX = int(detected_corners[2][0]) + xOffset
TopLeftCornerOfBoxY = int(detected_corners[2][1])
for i in range(1, 9):
    CroppedImage = SheetImage[TopLeftCornerOfBoxY + xThreshold: TopLeftCornerOfBoxY + RectangleBoxHeight,
                              TopLeftCornerOfBoxX + yThreshold + ((i - 1) * RectangleBoxWidth // 8):
                              TopLeftCornerOfBoxX + (i * (RectangleBoxWidth // 8))- xThreshold]
    cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\LN' 
                + str(i) + '.jpg', CroppedImage)
checkBoxThresh = 5
TopLeftCornerOfBoxX = int(detected_corners[3][0]) + xOffset
TopLeftCornerOfBoxY = int(detected_corners[3][1])
CroppedImage = SheetImage[TopLeftCornerOfBoxY + checkBoxThresh : TopLeftCornerOfBoxY + 18,
                          TopLeftCornerOfBoxX + checkBoxThresh : TopLeftCornerOfBoxX + 18]
cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\PHD' 
            + '.jpg', CroppedImage)

TopLeftCornerOfBoxX = int(detected_corners[4][0]) + xOffset
TopLeftCornerOfBoxY = int(detected_corners[4][1])
CroppedImage = SheetImage[TopLeftCornerOfBoxY + checkBoxThresh: TopLeftCornerOfBoxY + 18,
                          TopLeftCornerOfBoxX + checkBoxThresh : TopLeftCornerOfBoxX + 18]
cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\MS' 
            + '.jpg', CroppedImage)

TopLeftCornerOfBoxX = int(detected_corners[5][0]) + xOffset
TopLeftCornerOfBoxY = int(detected_corners[5][1])
CroppedImage = SheetImage[TopLeftCornerOfBoxY + checkBoxThresh : TopLeftCornerOfBoxY + 18,
                          TopLeftCornerOfBoxX + checkBoxThresh : TopLeftCornerOfBoxX + 18]
cv2.imwrite('C:\\Users\\mahdi\\OneDrive\\Desktop\\computer_vision\\project\\extracted_form\\BS' 
            + '.jpg', CroppedImage)
