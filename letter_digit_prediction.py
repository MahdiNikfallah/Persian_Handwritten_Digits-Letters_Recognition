import os
import numpy as np
import cv2
import glob
import time
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

expected_corners = [
        np.array([38, 188], np.float),  #  Student Id 
        np.array([37, 271], np.float),  #  First Name 
        np.array([37, 356], np.float),  #  Name 
        np.array([68, 457], np.float),  #  PHD 
        np.array([205, 456], np.float), #  MS 
        np.array([404, 455], np.float)  #  BC
    ]


farsi_digits_en = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'alef', 'be', 'pe', 'te', 'se', 'jim', 'che', 'he', 'khe', 'dal',
    'zal', 're', 'ze', 'zhe', 'sin', 'shin', 'sad', 'zad', 'ta', 'za',
    'ein', 'ghein', 'fe', 'ghaf', 'kaf', 'gaf', 'lam', 'mim', 'non', 'vav',
    'he', 'ye',
]

Alphabet = {10: 'ا', 11: 'ب', 12: 'پ', 13: 'ت', 14: 'ث', 15: 'ج', 16: 'چ', 17: 'ح', 18: 'خ',
            19: 'د', 20: 'ذ', 21: 'ر', 22: 'ز', 23: 'ژ', 24: 'س', 25: 'ش', 26: 'ص', 27: 'ض',
            28: 'ط', 29: 'ظ', 30: 'ع', 31: 'غ', 32: 'ف', 33: 'ق', 34: 'ک', 35: 'گ', 36: 'ل',
            37: 'م', 38: 'ن', 39: 'و', 40: 'ه', 41: 'ی'}

RectangleBoxHeight = 60
RectangleBoxWidth = 478
num_classes_letter = 32
num_classes_digit = 10
img_size = 64
draw_plot = False

form_path = ".\\phase1_dataset\\"
extracted_data_path = ".\\data\\"

fields_id = []
fields_firstname = []
fields_lastname = []

first_name = []
last_name = []
id_digits = []

fields_checks = [None, None, None]

PHD = False
MS = False
BC = False

#-------------------------------------------PHASE-1-------------------------------------

def correct_perspective(Image):
    Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    Corners, IDs, _ = cv2.aruco.detectMarkers(Image, Dictionary)
    Corners = np.array(Corners)

    Markers = {'TopLeft': np.reshape(Corners[IDs == 34], (4, 2)), 
               'TopRight': np.reshape(Corners[IDs == 35], (4, 2)),
               'BottomLeft': np.reshape(Corners[IDs == 33], (4, 2)), 
               'BottomRight': np.reshape(Corners[IDs == 36],(4, 2))}
    
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
        
    return SheetImage


def binary_image(GraySheetImage):

    BlurredSheetImage = cv2.GaussianBlur(GraySheetImage, (7, 7), 0)    
    
    '''
    cv2.imshow("test",SheetImage)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    
    
    # Remove Markers for Detecting Boxes
    
    
    
    th2 = cv2.adaptiveThreshold(BlurredSheetImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,19,2.2)
        
    th2 = cv2.bitwise_not(th2)
    kernelC = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th2 = cv2.morphologyEx(th2,cv2.MORPH_CLOSE, kernelC)
    
    return th2


def thin_image(img):
    
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape,dtype='uint8')
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img)!=0):
        # Erosion
        erode = cv2.erode(img,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img = erode.copy()
        
    thin = cv2.dilate(thin,kernel)
    
    return thin


def find_corners(img):
    BlockSize = 3
    SobelKernelSize = 3
    Alpha = 0.01
    
    HarrisScores = cv2.cornerHarris(img, BlockSize, SobelKernelSize, Alpha)
    HarrisScores = HarrisScores / HarrisScores.max()
    ThresholdedHarrisScores = np.uint8(HarrisScores > 0.2) * 255
    
    nC, CC, stats, centroids = cv2.connectedComponentsWithStats(ThresholdedHarrisScores)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(ThresholdedHarrisScores, np.float32(centroids), (5, 5), (-1, -1), criteria)
    
    return corners


def detect_boxes_corners(corners):
    box_corners = [None, None, None, None, None, None]
    for i in range(0, len(expected_corners)):
        tmp_corner = expected_corners[i]
        #cv2.circle(SheetImage, (int(tmp_corner[0]), int(tmp_corner[1])), 10, (255, 255, 255))
        dist = 999999
        for j in range(len(corners)):
            tempDist = np.linalg.norm(corners[j] - tmp_corner)
            if tempDist < dist:
                dist = tempDist
                box_corners[i] = corners[j]
                
    return box_corners
    '''
    for j in range(len(detected_corners)):
        cv2.circle(SheetImage, (int(detected_corners[j][0]), int(detected_corners[j][1])), 1, (255, 255, 255))
        '''
        
def detect_boxes(box_corners, GraySheetImage):
    TopLeftCornerOfBoxX = int(box_corners[0][0]) + 6
    TopLeftCornerOfBoxY = int(box_corners[0][1])
    xThreshold = 3
    yThreshold = 5
    xOffset = 8;
    edgeThresh = 115
    for i in range(1, 9):
        CroppedImage = GraySheetImage[TopLeftCornerOfBoxY + xThreshold: TopLeftCornerOfBoxY + RectangleBoxHeight,
                                  TopLeftCornerOfBoxX + yThreshold + ((i - 1) * RectangleBoxWidth // 8):
                                  TopLeftCornerOfBoxX + (i * (RectangleBoxWidth // 8)) - xThreshold + 3]
            
        tempImg = clearLines(CroppedImage)
        tempImg = CroppedImage[6:tempImg.shape[0]-6,6:tempImg.shape[1]-6]
        
        sobelx = cv2.Sobel(tempImg, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(tempImg, cv2.CV_64F, 0, 1, ksize=5)
        E = np.sqrt((sobelx*sobelx) + (sobely*sobely))
        '''
        cv2.imshow("sobel",tempImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(np.average(E))
        ''' 
        if np.average(E)>edgeThresh:
            #cv2.imwrite(extracted_data_path + 'ID' + str(9 - i) + '.jpg', CroppedImage)
            fields_id.append(CroppedImage)
    
    
    TopLeftCornerOfBoxX = int(box_corners[1][0]) + xOffset
    TopLeftCornerOfBoxY = int(box_corners[1][1])
    
    for i in range(1, 9):
        CroppedImage = GraySheetImage[TopLeftCornerOfBoxY + xThreshold: TopLeftCornerOfBoxY + RectangleBoxHeight,
                                  TopLeftCornerOfBoxX + yThreshold + ((i - 1) * RectangleBoxWidth // 8):
                                  TopLeftCornerOfBoxX + (i * (RectangleBoxWidth // 8))- xThreshold + 3]
        
            
        tempImg = clearLines(CroppedImage)
        tempImg = CroppedImage[6:tempImg.shape[0]-6,6:tempImg.shape[1]-6]
        
        
        sobelx = cv2.Sobel(tempImg, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(tempImg, cv2.CV_64F, 0, 1, ksize=5)
        E = np.sqrt((sobelx*sobelx) + (sobely*sobely))
        
        '''
        cv2.imshow("sobel",tempImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(np.average(E))
        '''
        if np.average(E)>edgeThresh:
            #cv2.imwrite(extracted_data_path + 'FN' + str(9 - i) + '.jpg', CroppedImage)
            fields_firstname.append(CroppedImage)
    
    TopLeftCornerOfBoxX = int(box_corners[2][0]) + xOffset
    TopLeftCornerOfBoxY = int(box_corners[2][1])
    for i in range(1, 9):
        CroppedImage = GraySheetImage[TopLeftCornerOfBoxY + xThreshold: TopLeftCornerOfBoxY + RectangleBoxHeight,
                                  TopLeftCornerOfBoxX + yThreshold + ((i - 1) * RectangleBoxWidth // 8):
                                  TopLeftCornerOfBoxX + (i * (RectangleBoxWidth // 8))- xThreshold + 3]
            
        tempImg = clearLines(CroppedImage)
        tempImg = CroppedImage[6:tempImg.shape[0]-6,6:tempImg.shape[1]-6]
        
        sobelx = cv2.Sobel(tempImg, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(tempImg, cv2.CV_64F, 0, 1, ksize=5)
        E = np.sqrt((sobelx*sobelx) + (sobely*sobely))
        '''
        cv2.imshow("sobel",tempImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(np.average(E))
        '''  
        if np.average(E)>edgeThresh:
            #cv2.imwrite(extracted_data_path + 'LN' + str(9 - i) + '.jpg', CroppedImage)
            fields_lastname.append(CroppedImage)
        
    checkBoxThresh = 5
    TopLeftCornerOfBoxX = int(box_corners[3][0]) + xOffset
    TopLeftCornerOfBoxY = int(box_corners[3][1])
    CroppedImage = GraySheetImage[TopLeftCornerOfBoxY + checkBoxThresh : TopLeftCornerOfBoxY + 15,
                              TopLeftCornerOfBoxX + checkBoxThresh : TopLeftCornerOfBoxX + 15]
    #cv2.imwrite(extracted_data_path + 'PHD' + '.jpg', CroppedImage)
    fields_checks[0] = CroppedImage
    
    TopLeftCornerOfBoxX = int(box_corners[4][0]) + xOffset
    TopLeftCornerOfBoxY = int(box_corners[4][1])
    CroppedImage = GraySheetImage[TopLeftCornerOfBoxY + checkBoxThresh: TopLeftCornerOfBoxY + 15,
                              TopLeftCornerOfBoxX + checkBoxThresh : TopLeftCornerOfBoxX + 15]
    #cv2.imwrite(extracted_data_path + 'MS' + '.jpg', CroppedImage)
    fields_checks[1] = CroppedImage
    
    TopLeftCornerOfBoxX = int(box_corners[5][0]) + xOffset
    TopLeftCornerOfBoxY = int(box_corners[5][1])
    CroppedImage = GraySheetImage[TopLeftCornerOfBoxY + checkBoxThresh : TopLeftCornerOfBoxY + 15,
                              TopLeftCornerOfBoxX + checkBoxThresh : TopLeftCornerOfBoxX + 15]
    #cv2.imwrite(extracted_data_path + 'BC' + '.jpg', CroppedImage)
    fields_checks[2] = CroppedImage


def detect_grade(fields_checks, sheet_image, corners):
    global BC 
    global MS
    global PHD
    threshold = 0.8
    corners = np.array(corners, int)
    box_med = np.median(fields_checks[0])
    if(box_med < np.median(sheet_image[corners[0][0]-23:corners[0][0]-17,
                                       corners[0][1]-23:corners[0][1]-17]) * threshold):
        PHD = True
    
    box_med = np.median(fields_checks[1])
    if(box_med < np.median(sheet_image[corners[1][0]-23:corners[1][0]-17,
                                       corners[1][1]-23:corners[1][1]-17]) * threshold):
        MS = True
        
    box_med = np.median(fields_checks[2])
    if(box_med < np.median(sheet_image[corners[2][0]-23:corners[2][0]-17,
                                       corners[2][1]-23:corners[2][1]-17]) * threshold ):
        BC = True

#-------------------------------------------PHASE-1-------------------------------------
#-------------------------------------------PHASE-2-------------------------------------
    
def clearLines(img):
    BlurredSheetImage = cv2.GaussianBlur(img, (9, 9), 0)
    th2 = cv2.adaptiveThreshold(BlurredSheetImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,1.9)

    r = 0
    
    kernelC = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    th2 = cv2.morphologyEx(th2,cv2.MORPH_OPEN, kernelC)
    
    kernelC = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    th2 = cv2.morphologyEx(th2,cv2.MORPH_OPEN, kernelC)

    cv2.destroyAllWindows()
    for i in range(img.shape[1],img.shape[1]-20,-1):
        av = np.average(th2[:,i-1])
        if(r==0 and av< 50):
            r=1;
        elif(r==1 and av > 205):
            img = img[:,0:i-1]
            th2 = th2[:,0:i-1]
            break;
    r = 0
    for i in range(0, 20):
        av = np.average(th2[:,i])
        if(r==0 and av< 50):
            r=1;
        elif(r==1 and av > 205):
            img = img[:,i+1:]
            th2 = th2[:,i+1:]
            break;
    r = 0
    for i in range(0, 20):
        av = np.average(th2[i,:])
        if(r==0 and av< 50):
            r=1;
        elif(r==1 and av > 205):
            img = img[i+1:,:]
            th2 = th2[i+1:,:]
            break;
    r = 0
    for i in range(img.shape[0],img.shape[0]-20,-1):
        av = np.average(th2[i-1,:])
        if(r==0 and av< 50):
            r=1;
        elif(r==1 and av > 205):
            img = img[0:i-1,:]
            th2 = th2[0:i-1,:]
            break;
    return img
    

def build_model_letter(inputs):
  x = inputs

  x = Conv2D(filters=16, kernel_size=(5, 5), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

  x = Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x) 

  x = Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x) 

  x = Flatten()(x)

  x = Dropout(0.2)(x)

  x = Dense(128, activation="relu")(x)

  x = Dropout(0.1)(x)

  x = Dense(128, activation="relu")(x)

  x = Dense(64, activation = 'relu')(x)

  outputs = Dense(num_classes_letter, activation="softmax")(x)

  model = Model(inputs, outputs, name="LeNet")
  model.summary()
  
  return model



def build_model_digit(inputs):
  x = inputs

  x = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

  x = Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)  


  x = Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x) 
  

  x = Flatten()(x)


  x = Dense(64, activation="relu")(x)

  x = Dropout(0.2)(x)

  x = Dense(32, activation = 'relu')(x)

  outputs = Dense(num_classes_digit, activation="softmax")(x)

  model = Model(inputs, outputs, name="LeNet")
  model.summary()
  
  return model


'''
def build_model_digit(inputs):
  x = inputs

  x = Conv2D(filters=16, kernel_size=(5, 5), padding="valid", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

  x = Conv2D(filters=32, kernel_size=(5, 5), padding="valid", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x) 

  x = Conv2D(filters=64, kernel_size=(5, 5), padding="valid", activation="relu")(x)
  x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x) 

  x = Flatten()(x)

  x = Dense(128, activation="relu")(x)

  x = Dropout(0.2)(x)

  x = Dense(64, activation = 'relu')(x)

  outputs = Dense(num_classes_digit, activation="softmax")(x)

  model = Model(inputs, outputs, name="LeNet")
  model.summary()
  
  return model
'''


def resized_image(output_img):
    
    med = np.median(output_img)
    
    #output_img = output_img[2:output_img.shape[0]-2,2:output_img.shape[1]-2]
    
    xLength = output_img.shape[1]
    yLength = output_img.shape[0]
    
    if(xLength > yLength):
        newYLength = int(yLength * img_size/xLength)
        resizeDim = (img_size, newYLength)
        resized = cv2.resize(output_img, resizeDim, interpolation = cv2.INTER_AREA)
        
        if(newYLength % 2 == 0):    
            output_img = cv2.copyMakeBorder(resized, int((img_size - newYLength)/2),
                                        int((img_size - newYLength)/2), 0, 0, cv2.BORDER_CONSTANT, None, med)
        else:
            output_img = cv2.copyMakeBorder(resized, (int((img_size - newYLength-1)/2))+1,
                                        int((img_size - newYLength-1)/2), 0, 0, cv2.BORDER_CONSTANT, None, med)
                
    
    else:
        newXLength = int(xLength * img_size/yLength)
        resizeDim = (newXLength, img_size)
        resized = cv2.resize(output_img, resizeDim, interpolation = cv2.INTER_AREA)
        
        if(newXLength % 2 == 0):    
            output_img = cv2.copyMakeBorder(resized, 0,
                                        0, int((img_size - newXLength)/2), int((img_size - newXLength)/2),
                                        cv2.BORDER_CONSTANT, None, med)
        else:
            output_img = cv2.copyMakeBorder(resized,0,0, (int((img_size - newXLength-1)/2))+1,
                                        int((img_size - newXLength-1)/2), cv2.BORDER_CONSTANT, None, med)
            
    return output_img

def build_and_load_models():
    
    input = Input((img_size, img_size, 1))
    model_letter = build_model_letter(input)
    model_letter.load_weights("model_letters_2.h5")
    
    input = Input((img_size, img_size, 1))
    model_digit = build_model_digit(input)
    model_digit.load_weights("model_digit_6.h5")
    
    return model_digit, model_letter

def letter_detection(name, model, flag, file_name):
    
    string_detected = ''
    
    #file_names = glob.glob(extracted_data_path + name + '.jpg')
    if draw_plot:
        plt.figure(figsize=(3, 3))
    if flag == 0:
        fields = fields_firstname
    else:
        fields = fields_lastname
    for i in range(0,len(fields)):
        img = fields[i]
        #img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        img = clearLines(img)
        img = resized_image(img)
        orig_img = np.copy(img)
        img = img_to_array(img) / 255.
        img = np.expand_dims(img, 0)
        #st = time.time()
        predictions = model.predict(img)[0]
        #end = time.time()
        #print("let predict time : " + str(end - st))
        label = np.argmax(predictions)
        proba = np.max(predictions)
        label = np.argmax(predictions)
        proba = np.max(predictions)
        output = cv2.resize(orig_img, (300, 300))
        string_detected+= Alphabet[label + 10]
        if draw_plot:
            plt.subplot(3, 3, i + 1)
            plt.imshow(output, cmap="gray")
            plt.axis("off")
            plt.title("{}: {:.2f}%".format(farsi_digits_en[10 + label], proba * 100),  fontsize=8)
            if(file_name!=None):
                plt.savefig(".//plots//training_plot" + str(file_name).replace('.//inputs\\', '')  + ".png")
    
    return string_detected
        
    
        
def digit_detection(name, model, file_name):
    
    string_detected = ''
    
    #file_names = glob.glob(extracted_data_path + 'ID*.jpg')
    
    if draw_plot:
        plt.figure(figsize=(3, 3))
    for i in range(0, len(fields_id)):
        img = fields_id[i]
        #img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        img = clearLines(img)
        img = resized_image(img)
        orig_img = np.copy(img)
        img = img_to_array(img) / 255.
        img = np.expand_dims(img, 0)
        predictions = model.predict(img)[0]
        label = np.argmax(predictions)
        proba = np.max(predictions)
        label = np.argmax(predictions)
        proba = np.max(predictions)
        output = cv2.resize(orig_img, (300, 300))
        string_detected+= str(label)
        if draw_plot:
            plt.subplot(3, 3, i + 1)
            plt.imshow(output, cmap="gray")
            plt.axis("off")
            plt.title("{}: {:.2f}%".format(label, proba * 100),fontsize=8)
            if(file_name!=None):
                plt.savefig(".//plots//training_plot" + str(file_name).replace('.//inputs\\', '') + ".png")
            
    return string_detected

#---------------------------------------------------PHASE-2-----------------------------
#---------------------------------------------------PHASE-1-----------------------------

model_digit, model_letter = build_and_load_models()



resizeDim = (img_size, img_size)

file_names = glob.glob('.//inputs//*.jpg')
resized = cv2.resize(cv2.imread(".//inputs//1.jpg",cv2.IMREAD_GRAYSCALE), resizeDim, interpolation = cv2.INTER_AREA)
fields_firstname.append(resized)
fields_id.append(resized)
letter_detection('FN*', model_letter, 0,None)
digit_detection('ID*', model_digit,None)
fields_firstname = []
fields_id = []

draw_plot = True

for i,filename in enumerate(file_names):
    
    Image = cv2.imread(filename)
    
    start_time = time.time()
    
    sheet_image = correct_perspective(Image)
    
    GraySheetImage = cv2.cvtColor(sheet_image, cv2.COLOR_BGR2GRAY)
    
    binary_sheet = binary_image(GraySheetImage)
    
    binary_sheet_thin = thin_image(binary_sheet)
    
    corners = find_corners(binary_sheet_thin)
    
    box_corners = detect_boxes_corners(corners)
    
    detect_boxes(box_corners, GraySheetImage)
    
    detect_grade(fields_checks, sheet_image, corners)
        
    #---------------------------------------------------PHASE-1-----------------------------
    #---------------------------------------------------PHASE-2-----------------------------
    
    print("3333333333 + " + filename)
    
    id_digits = digit_detection('ID*', model_digit, filename[0:len(filename) - 4] + "_0")
    
    first_name = letter_detection('FN*', model_letter, 0 , filename[0:len(filename) - 4] + "_1")
    
    last_name = letter_detection('LN*', model_letter, 1, filename[0:len(filename) - 4] + "_2")
    
    first_name = first_name[::-1]
    last_name = last_name[::-1]
    
    end_time = time.time()
    
    print("elapsed time : " + str((end_time - start_time)))
    
    #id_digits = id_digits[::-1]
    
    print("=================^_^===================" )
    
    print("id is : " +  id_digits )
    
    print("First Name : " + first_name)
    
    print("Last Name : " + last_name)
    
    if draw_plot:
        plt.figure(figsize=(3, 3))
        plt.subplot(3, 3, 1)
        plt.imshow(fields_checks[0], cmap="gray")
        plt.axis("off")
        plt.title("PHD")
        plt.subplot(3, 3, 2)
        plt.imshow(fields_checks[1], cmap="gray")
        plt.axis("off")
        plt.title("MS")
        plt.subplot(3, 3, 3)
        plt.imshow(fields_checks[2], cmap="gray")
        plt.axis("off")
        plt.title("BC")
    
    
    if(MS):
        print("Grade : کارشناسی ارشد")
    elif(PHD):
        print("Grade : دکتری")
    else:
        print("Grade : کارشناسی")
        
    fields_id = []
    fields_firstname = []
    fields_lastname = []
    
    first_name = []
    last_name = []
    id_digits = []
    
    fields_checks = [None, None, None]
    
    PHD = False
    MS = False
    BC = False
        
    print("next -----------")
    
    

#---------------------------------------------------PHASE-2-----------------------------

'''
tempp = np.copy(SheetImage)
for j in range(len(corners)):
    cv2.circle(tempp, (int(corners[j][0]), int(corners[j][1])), 1, (255, 255, 255))
'''

