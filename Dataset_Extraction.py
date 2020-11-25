import glob
import os
import random

import cv2
import numpy as np
import six.moves.cPickle as pickle

output_folder = './out/'
dataset_folder = './dataset/'
img_height = 1490
img_width = 1049
max_output_size = 75

RED = '\033[91m'
GREEN = '\033[92m'
BOLD = '\033[1m'

box_size = (1049 / 14, 1490 / 21)
hough_min_votes = 50
hough_distance_resolution = 1
hough_angle_resolution = np.pi / 20
find_line_trh = 10

dataD = []

farsi_digits = [
    '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
    'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
    'ه', 'ی',
]

page_one_digit = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 2, 3]
page_two_digit = [4, 5, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 6, 7, 8, 9]

output_dataset = []
output_dataset_letter = []
output_dataset_digit = []

s_id_to_file = {}


def findAruco(img, one_or_two=True):
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    marker_ids = marker_ids.tolist()
    a_30_i = marker_ids.index([30])
    a_31_i = marker_ids.index([31])
    a_33_i = marker_ids.index([33])
    a_32_i = marker_ids.index([32])

    p1 = int(marker_corners[a_30_i][0][0][0]), int(marker_corners[a_30_i][0][0][1])
    p2 = int(marker_corners[a_31_i][0][1][0]), int(marker_corners[a_31_i][0][1][1])
    p3 = int(marker_corners[a_33_i][0][2][0]), int(marker_corners[a_33_i][0][2][1])
    p4 = int(marker_corners[a_32_i][0][3][0]), int(marker_corners[a_32_i][0][3][1])

    return np.array([p1, p2, p3, p4], dtype=np.float32) if one_or_two else np.array([p3, p4, p1, p2], dtype=np.float32)


def clearLines(img):
    BlurredSheetImage = cv2.GaussianBlur(img, (9, 9), 0)
    th2 = cv2.adaptiveThreshold(BlurredSheetImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,1.9)
    #ret, _ = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print("otsu" , ret)
    #print("right")
    r = 0
    
    kernelC = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    th2 = cv2.morphologyEx(th2,cv2.MORPH_OPEN, kernelC)
    
    kernelC = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    th2 = cv2.morphologyEx(th2,cv2.MORPH_OPEN, kernelC)
    
    '''
    cv2.imshow('th',th2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if(cv2.waitKey(0)==113):
        break;
    '''
    cv2.destroyAllWindows()
    for i in range(img.shape[1],img.shape[1]-20,-1):
        av = np.average(th2[:,i-1])
        #print(np.average(img[:,i-1]))
        if(r==0 and av< 50):
            r=1;
            temp = av
            #print("t1 triggered")
        elif(r==1 and av > 205):
            img = img[:,0:i-1]
            th2 = th2[:,0:i-1]
            #print("crop")
            break;
    #print("left")
    r = 0
    for i in range(0, 20):
        av = np.average(th2[:,i])
        #print(av)
        if(r==0 and av< 50):
            r=1;
            temp = av
        elif(r==1 and av > 205):
            img = img[:,i+1:]
            th2 = th2[:,i+1:]
            break;
    #print("top")
    r = 0
    for i in range(0, 20):
        av = np.average(th2[i,:])
        #print(np.average(img[i,:]))
        if(r==0 and av< 50):
            r=1;
            temp = av
        elif(r==1 and av > 205):
            img = img[i+1:,:]
            th2 = th2[i+1:,:]
            break;
    #print("bottom")
    r = 0
    for i in range(img.shape[0],img.shape[0]-20,-1):
        av = np.average(th2[i-1,:])
        #print(np.average(av))
        if(r==0 and av< 50):
            r=1;
            temp = av
            #print("t1 triggered")
        elif(r==1 and av > 205):
            img = img[0:i-1,:]
            th2 = th2[0:i-1,:]
            #print("crop")
            break;
    #print(x/len(file_names))
    return img



def correctPerspective(img, aruco_positions, correct_form, output_size):
    H = cv2.getPerspectiveTransform(aruco_positions, correct_form)
    J = cv2.warpPerspective(img, H, output_size)
    return J


def doPerspectiveAndSaveRow(aruco_positions, img, filename):
    x = 5
    n = int(210 * x)
    m = int(297 * x)
    output_size = (n, m)
    correct_form = np.array([
        (0, 0), (n, 0),
        (n, m), (0, m)
    ], dtype=np.float32)

    img = correctPerspective(img, aruco_positions, correct_form, output_size)
    output_filename = out_folder + 'row/' + filename.split('/')[-1]
    cv2.imwrite(output_filename, img)

    return img


def findXAndYLines(img):
    G = img
    E = cv2.Canny(G, 80, 100)

    L = cv2.HoughLines(E, hough_distance_resolution, hough_angle_resolution, hough_min_votes)
    all_x_lines = []
    all_y_lines = []
    for [[rho, theta]] in L:
        if -0.01 < theta / np.pi < 0.01:
            all_x_lines.append(np.cos(theta) * rho)

        if 0.49 < theta / np.pi < 0.51:
            all_y_lines.append(np.sin(theta) * rho)

    x_founded = []
    for i in range(1, 14):
        tmp = []
        for j in all_x_lines:
            if i * box_size[0] - find_line_trh < float(j) < i * box_size[0] + find_line_trh:
                tmp.append(int(j))
        x_founded.append(tmp)

    y_founded = []
    for i in range(1, 21):
        tmp = []
        for j in all_y_lines:
            if i * box_size[1] - find_line_trh < float(j) < i * box_size[1] + find_line_trh:
                tmp.append(int(j))
        y_founded.append(tmp)

    output_x_lines = []
    output_y_lines = []
    counter = 0
    for j in x_founded:
        counter += 1
        size = len(j)
        mean = 0
        if size != 0:
            for i in range(len(j)):
                mean += j[i]
            output_x_lines.append(mean / size)
        else:
            output_x_lines.append(int(counter * box_size[0]))
    counter = 0
    for j in y_founded:
        counter += 1
        size = len(j)
        mean = 0
        if size != 0:
            for i in range(len(j)):
                mean += j[i]
            output_y_lines.append(mean / size)
        else:
            output_y_lines.append(int(counter * box_size[1]))
    return output_x_lines, output_y_lines


def saveTile(img, one_or_two, line, id, last_start, last_end, s_id):
    ch = 0;
    name = ''
    if one_or_two:
        name = page_one_digit[line]
    else:
        name = page_two_digit[line]
    output_file_name = out_folder + 'extracted/{}/{}.png'.format(name, id)

    x_size = int(last_end[0]) - int(last_start[0])
    y_size = int(last_end[1]) - int(last_start[1])

    # x_offset = max_output_size - x_size
    # y_offset = max_output_size - y_size
    # np.ceil(x_offset / 2)
    output_img = img[int(last_start[1]):int(last_end[1]), int(last_start[0]):int(last_end[0])]
    output_img = clearLines(output_img)
    #output_img = cv2.resize(output_img, (max_output_size, max_output_size), None, interpolation=cv2.INTER_CUBIC)
    #output_img = output_img[5:max_output_size - 5, 5:max_output_size - 5]
    
    
    xLength = int(last_end[0]) - int(last_start[0])
    yLength = int(last_end[1]) - int(last_start[1])
    
    med = np.median(output_img)
    
    output_img = output_img[2:output_img.shape[0]-2,2:output_img.shape[1]-2]
    
    if(xLength > yLength):
        newYLength = int(yLength * 64/xLength)
        resizeDim = (64, newYLength)
        resized = cv2.resize(output_img, resizeDim, interpolation = cv2.INTER_AREA)
        
        if(newYLength % 2 == 0):    
            output_img = cv2.copyMakeBorder(resized, int((64 - newYLength)/2),
                                        int((64 - newYLength)/2), 0, 0, cv2.BORDER_CONSTANT, None, med)
        else:
            output_img = cv2.copyMakeBorder(resized, (int((64 - newYLength-1)/2))+1,
                                        int((64 - newYLength-1)/2), 0, 0, cv2.BORDER_CONSTANT, None, med)
            

    else :
        newXLength = int(xLength * 64/yLength)
        resizeDim = (newXLength, 64)
        resized = cv2.resize(output_img, resizeDim, interpolation = cv2.INTER_AREA)
        if(newXLength % 2 == 0):    
            output_img = cv2.copyMakeBorder(resized, 0,
                                        0, int((64 - newXLength)/2), int((64 - newXLength)/2),
                                        cv2.BORDER_CONSTANT, None, med)
        else:
            output_img = cv2.copyMakeBorder(resized,0,0, (int((64 - newXLength-1)/2))+1,
                                        int((64 - newXLength-1)/2), cv2.BORDER_CONSTANT, None, med)
            
    # cv2.imshow('corners', output_img)
    # cv2.waitKey(0)  # press any key
    if(int(name) > 9):
        output_dataset_digit.append([name, output_img])
    else:
        output_dataset_letter.append([name, output_img])
    
    cv2.imwrite(output_file_name, output_img)


def dump_to_pkl():

    
    data_digit_size = len(output_dataset_digit)
    data_letter_size = len(output_dataset_letter)
    
    data = {}
    
    data['digit'] = {}
    data['digit']['train'] = {}
    
    data['digit']['train']['data'] = []
    data['digit']['train']['target'] = []
    data['digit']['val'] = {}
    data['digit']['val']['data'] = []
    data['digit']['val']['target'] = []
    data['digit']['test'] = {}
    data['digit']['test']['data'] = []
    data['digit']['test']['target'] = []
    
    data['letter'] = {}
    data['letter']['train'] = {}
    data['letter']['train']['data'] = []
    data['letter']['train']['target'] = []
    data['letter']['val'] = {}
    data['letter']['val']['data'] = []
    data['letter']['val']['target'] = []
    data['letter']['test'] = {}
    data['letter']['test']['data'] = []
    data['letter']['test']['target'] = []
    
    train_digit_size = int((data_digit_size * 80) / 100)
    val_digit_size = int((data_digit_size * 15) / 100)
    
    train_letter_size = int((data_letter_size * 80) / 100)
    val_letter_size = int((data_letter_size * 15) / 100)
    
    train_data = []

    for i in range(0, train_digit_size):
        train_data.append(output_dataset_digit[i])
        
    random.shuffle(train_data)
    
    for i in range(0, train_digit_size):
        data['digit']['train']['data'].append(train_data[i][1])
        data['digit']['train']['target'].append(train_data[i][0])
    
    val_data = []
    
    for i in range(train_digit_size, train_digit_size + val_digit_size):
        val_data.append(output_dataset_digit[i])
    
    random.shuffle(val_data)
    
    for i in range(0, len(val_data)):
        data['digit']['val']['data'].append(val_data[i][1])
        data['digit']['val']['target'].append(val_data[i][0])
        
        
    
    for i in range(train_digit_size + val_digit_size, data_digit_size):
        data['digit']['test']['data'].append(output_dataset_digit[i][1])
        data['digit']['test']['target'].append(output_dataset_digit[i][0])
         
    train_data = []

    for i in range(0, train_letter_size):
        train_data.append(output_dataset_letter[i])
        
    random.shuffle(train_data)
        
    
    for i in range(0, train_letter_size):
        data['letter']['train']['data'].append(train_data[i][1])
        data['letter']['train']['target'].append(train_data[i][0])
    
    val_data = []
    
    for i in range(train_letter_size, train_letter_size + val_letter_size):
        val_data.append(output_dataset_letter[i])
        
    random.shuffle(val_data)
    
    for i in range(0, len(val_data)):
        data['letter']['val']['data'].append(val_data[i][1])
        data['letter']['val']['target'].append(val_data[i][0])
        
    for i in range(train_letter_size + val_letter_size, data_letter_size):
        data['letter']['test']['data'].append(output_dataset_letter[i][1])
        data['letter']['test']['target'].append(output_dataset_letter[i][0])
        
        
    DataD = data
    print(GREEN, '🗂 -> ', 'Dump to PKL File')
    pickle.dump(data, open('farsi_handwritten_64.pkl', 'wb', -1))


def main():
    file_names = glob.glob(dataset_folder + '*.jpg')
    counter = 0
    id_counter = 0
    all_files_count = len(file_names)
    bad_files = 0
    delete_these = []
    for filename in file_names:
        counter += 1
        img = cv2.imread(filename, 0)
        print()
        print(GREEN, BOLD, '----------------------------------------------------')
        percent = int((counter * 100) / all_files_count)
        print(BOLD, '{}% {}/{} - {}'.format(percent, counter, all_files_count, filename))

        try:
            file_type = filename.split('/')[-1].split('_')[-1][0]
            s_id = filename.split('/')[-1].split('_')[0]
            file_type = True if file_type == '1' else False
            aruco_positions = findAruco(img)
        except Exception as e:
            print(RED, '✖ ️-> ', 'cant find all aruco in {} DELETE'.format(filename))
            delete_these.append(filename)
            bad_files += 1
            continue

        img = doPerspectiveAndSaveRow(aruco_positions, img, filename)
        print(GREEN, '✔️ -> ', 'Row Image Exported.')

        # founded_corners, nC = findCorners(img)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #
        # for i in founded_corners:
        #     cv2.circle(img, (i[0], i[1]), 1, (0, 0, 255), thickness=-1)

        # Extract Tiles

        try:
            x_lines, y_lines = findXAndYLines(img)
        except Exception as e:
            print(RED, '✖ ️-> ', 'cant find all lines in {}'.format(filename))
            delete_these.append(filename)
            bad_files += 1
            continue

        for i in range(len(x_lines) + 1):
            y_start = 0
            y_end = len(y_lines) + 1

            if i <= 1:
                y_start = 2
                y_end = len(y_lines) - 1
            if i >= len(x_lines) - 1:
                y_start = 2
                y_end = len(y_lines) - 1

            for j in range(y_start, y_end):
                start = [0, 0]
                end = [0, 0]

                if i > 0:
                    start[0] = x_lines[i - 1]
                if j > 0:
                    start[1] = y_lines[j - 1]

                if i == len(x_lines):
                    end[0] = img_width
                else:
                    end[0] = x_lines[i]

                if j == len(y_lines):
                    end[1] = img_height
                else:
                    end[1] = y_lines[j]

                id_counter += 1
                try:
                    saveTile(img, file_type, j, id_counter, start, end, s_id)
                    if s_id not in s_id_to_file:
                        s_id_to_file[s_id] = [[id_counter, filename]]
                    else:
                        s_id_to_file[s_id].append([id_counter, filename])
                except Exception as ee:
                    print(RED, '✖ ️-> ', 'Save failed ({},{})'.format(i, j))
                    continue

        print(GREEN, '✔️ -> ', 'Tiles exported')

    print(GREEN, BOLD, '----------------------------------------------------')
    print(GREEN, '🗂 -> ', '{} Files | {} Bad | {} Data' \
          .format(all_files_count, bad_files, id_counter))

    dump_to_pkl()

    import json
    json = json.dumps(s_id_to_file)
    f = open("id_to_file_map.json", "w")
    f.write(json)
    f.close()

    if len(delete_these) > 0:
        print('sudo rm {}'.format(' '.join(delete_these)))


if __name__ == '__main__':

    # Make Output Folder
    out_folder = os.path.join(output_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(out_folder + 'row'):
        os.mkdir(out_folder + 'row')
    if not os.path.exists(out_folder + 'extracted'):
        os.mkdir(out_folder + 'extracted')
    for i in range(42):
        if not os.path.exists(out_folder + 'extracted/' + str(i)):
            os.mkdir(out_folder + 'extracted/' + str(i))

    main()