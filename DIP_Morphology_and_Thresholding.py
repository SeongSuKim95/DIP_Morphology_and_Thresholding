import numpy as np
from numpy import pi, exp, sqrt
import cv2
from matplotlib import pyplot as plt

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft2, ifft2

def main():

    img= cv2.imread('Fig1039(a)(polymersomes).tif',cv2.IMREAD_GRAYSCALE)
    histogram(img)
    cv2.imwrite('1039(d).jpg',Otsu_thresholding(img))
    cv2.imwrite('1039(c).jpg',Basic_global_thresholding(img,0))
    cv2.waitKey(0)
    #cv2.imshow('c',Basic_global_thresholding(img,0))
    #cv2.imwrite('1046(c).jpg',Basic_global_thresholding(img,0))
    #cv2.waitKey(0)
    #cv2.imshow('d',Otsu_thresholding(img))
    #cv2.imwrite('1046(d).jpg',Otsu_thresholding(img))
    #cv2.waitKey(0)

    #Boudary_img = Boundary_Extraction(img)
    #cv2.imshow('1',Boudary_img)
    #cv2.imwrite('9.14(b).jpg',Boudary_img)
    #cv2.waitKey(0)


    #Erosion_img = Erosion(img,3)
    #cv2.imshow('1',Erosion_img)
    #cv2.imwrite('9.11(b).jpg',Erosion_img)
    #cv2.waitKey(0)

    #Open_img = Dilation_box(Erosion_img)
    #cv2.imshow('2',Open_img)
    #cv2.imwrite('9.11(c).jpg',Open_img)
    #cv2.waitKey(0)

    #Open_Dilation_img = Dilation_box(Open_img)
    #cv2.imshow('3',Open_Dilation_img)
    #cv2.imwrite('9.11(d).jpg',Open_Dilation_img)
    #cv2.waitKey(0)

    #Open_Dilation_Erosion_img = Erosion(Open_Dilation_img,3)
    #cv2.imshow('4',Open_Dilation_Erosion_img)
    #cv2.imwrite('9.11(e).jpg',Open_Dilation_Erosion_img)
    #cv2.waitKey(0)

    #Moving_aver = Moving_average(img,30,0.1)
    #Otsu = Otsu_thresholding(img)
    #cv2.imshow('Otsu',Otsu)
    #cv2.imwrite('1050(b)Otsu.jpg',Otsu)
    #cv2.imshow('Moving',Moving_aver)
    #cv2.imwrite('1050(c)Moving_average n=30 b=0.1.jpg',Moving_aver)


    #result = local_threshold(img)
    #cv2.imshow('img',result)
    #cv2.imwrite('10.48(d)local_thresh.jpg',result)

    #histogram(img)
    #img = Multiple_threshold(img)
    #cv2.imshow('Partitioned',img)
    #cv2.imwrite('10.48(b)Multiple_threshold.jpg',img)
    #cv2.waitKey(0)

    #Otsu = Otsu_thresholding(img)
    #cv2.imshow('Otsu', Otsu)
    #cv2.waitKey(0)

    #cv2.imwrite('10.43(c)_Otsu.jpg',Otsu)

    #Laplacian = S_Laplacian_filter(img)
    #Percentile = percentile_threshold(Laplacian,99.5)
    #cv2.imshow('Percentile',Percentile)
    #cv2.waitKey(0)
    #cv2.imwrite('10.43(d)_99.5.jpg',Percentile)
    #img3 = img * Percentile
    #histogram(img3)
    #img4 = Global_thresholding(img,Otsu_edge_thresholding(img3))
    #cv2.imshow('10.43(e)',img4)
    #cv2.waitKey(0)
    #cv2.imwrite('10.43(e)k=112.jpg',img4)

    #Sobel = Sobel_abs(img)
    #Laplacian = S_Laplacian_filter(img)
    #Percentile_image = percentile_threshold(Sobel,99.9)
    #cv2.imshow('Sobel',Sobel)
    #cv2.imwrite('1042(c)Sobel.jpg',Sobel)
    #cv2.waitKey(0)
    #cv2.imshow('Percentile_image',Percentile_image)
    #cv2.imwrite('1042(c)Sobel_997.jpg',Percentile_image)
    #cv2.waitKey(0)
    #img3= img*Percentile_image
    #cv2.imshow('1042(d)Sobel.img',img3)
    #cv2.waitKey(0)
    #histogram(img3)
    #Otsu = Global_thresholding(img,Otsu_edge_thresholding(img3))

    #cv2.imshow('1042(f)',Otsu)
    #cv2.waitKey(0)
    #cv2.imwrite('1042(f)_Otsu.jpg',Otsu)

    #histogram(img)

    #Otsu = Otsu_thresholding(img)

    #cv2.imshow('Otsu',Otsu)
    #cv2.imwrite('1041(c)small_otsu.jpg',Otsu)
    #cv2.waitKey(0)

    #img2 = S_smoothing_linear(img,5)
    #cv2.imshow('5x5 Smoothing',img2)
    #cv2.waitKey(0)
    #cv2.imwrite('1041(d)5x5Smoothing.jpg',img2)


    #histogram(img2)
    #Otsu2 = Otsu_thresholding(img2)
    #cv2.imshow('Otsu2',Otsu2)
    #cv2.waitKey(0)
    #cv2.imwrite('1041(f)Otsu2.jpg',Otsu2)

    #GT= Basic_global_thresholding(img,0)
    #cv2.imshow('Global_Thresholding',GT)
    #cv2.waitKey(0)

    #img_ramp = cv2.imread('Fig1037(b)(intensity_ramp).tif',cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('Ramp',img_ramp)
    #cv2.waitKey(0)

    # img3= Intensity_ramp(img,img_ramp)
    # cv2.imshow('result',img3)
    # Histogram
    #histogram(img)
    #histogram(img_ramp)
    #histogram(img3)


    return

#CHAP 9
def Erosion(img,c):

    y,x = img.shape
    B = np.ones((c,c))

    img = np.array(img/255,dtype = float)

    zp = int((c - 1) / 2)
    result = np.zeros((y, x))
    image_reflect_padding = np.pad(img,zp,'constant',constant_values=0)

    for i in range(y):
        for j in range(x):
                if np.sum(image_reflect_padding[i:i+c,j:j+c]*B) == c**2 :
                    result[i][j] = 255
                else:
                    result[i][j] = 0

    result = np.array(result,dtype = np.uint8)
    # y, x = img.shape
    # for i in range(y):
    #  for j in range(x):
    #     if img[i][j] == 222 :
    #        print(1)
    return result

def Dilation_box(img):

    y, x = img.shape
    img = np.array(img, dtype=np.uint8)

    result = np.zeros((y, x))
    result = np.pad(result, 1, 'constant', constant_values=0)

    for i in range(y):
        for j in range(x):
            if img[i][j] == 255:
                with np.errstate(divide='ignore', invalid='ignore'):
                    result[i-1: i+2, j-1: j+2] = 255

    result = np.array(result, dtype=np.uint8)
    # y, x = img.shape
    # for i in range(y):
    #  for j in range(x):
    #     if img[i][j] == 222 :
    #        print(1)
    return result

def Dilation_disk(img):

    y,x = img.shape
    img = np.array(img,dtype = np.uint8)

    result = np.zeros((y, x))
    result = np.pad(result,1,'constant',constant_values=0)

    for i in range(y):
        for j in range(x):
                if img[i][j] == 255 :
                    with np.errstate(divide='ignore', invalid='ignore'):
                        result[i-1,j]=255
                        result[i+1,j]=255
                        result[i,j-1:j+2] = 255


    result = np.array(result,dtype = np.uint8)
    # y, x = img.shape
    # for i in range(y):
    #  for j in range(x):
    #     if img[i][j] == 222 :
    #        print(1)
    return result

def Boundary_Extraction(img):
    result = img - Erosion(img,3)
    return result
#CHAP 10
def histogram(img):

    #x = img.flatten()
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Frequency")
    plt.title("Histogram")
    plt.hist(img.ravel(),255,[1,256],density=True)
    plt.show()
    return

def Intensity_ramp(img,img_ramp):

    img = np.array(img,dtype = np.uint8)
    img_ramp = np.array(img_ramp,dtype=float)
    max = np.amax(img_ramp)
    img_ramp = (1/max)*img_ramp
    result = np.around(img * img_ramp)
    result = np.array(result,dtype =np.uint8)
    return result

def Global_thresholding(img,k_opt):
    img = img > k_opt
    img = 255*(img.astype(np.int))
    img = np.array(img, dtype=np.uint8)

    return img


def Basic_global_thresholding(img,c):

    img = np.array(img,dtype = np.uint8)

    T = np.mean(img)
    T_old = T
    T_new = T

    while abs(T_new - T_old) > c or T_new == T_old == T :
        T_old = T_new
        condition1 = img > T_old
        condition2 = img < T_old

        G1 = img[condition1]
        m1 = np.mean(G1)

        G2 = img[condition2]
        m2 = np.mean(G2)

        T_new =(m1+ m2)/2

    img = img > T_new
    img = 255*(img.astype(np.int))
    img = np.array(img,dtype = np.uint8)


    return img

def Otsu_thresholding(img):

    img = np.array(img,dtype = np.uint8)
    Histogram,bin_edges = np.histogram(img,bins=np.arange(257),density= True)
    Var_b_array = np.zeros((1,256))
    for k in range(256):
        with np.errstate(divide='ignore', invalid='ignore'):

            P1=Histogram[0:k]
            P2=Histogram[k:]

            C1 = Histogram[0:k]*np.arange(k)
            C2 = Histogram[k:] * (k + np.arange(256 - k))

            M1 = (1/P1.sum())*C1.sum()
            M2 = (1/P2.sum())*C2.sum()

            a=P1.sum()
            b=P2.sum()

            Var_b = a*b*(M1-M2)**2
            Var_b_array[0][k]= Var_b

    Var_b_array = np.nan_to_num(Var_b_array)
    Mg_array = Histogram * np.arange(256)
    Mg = Mg_array.sum()
    Var_g = np.var(img)

    k_opt = np.argmax(Var_b_array)
    Sep = Var_b_array[0][k_opt]/Var_g

    img = img > k_opt
    img = 255*(img.astype(np.int))
    img = np.array(img, dtype=np.uint8)

    #Var_g_ex = (np.square((np.arange(256) - Mg))*Histogram).sum()
    #x = a*M1 + b*M2 - Mg
    #print(np.amin(img))
    #print(np.amax(img))
    #print(Histogram.sum())

    # y, x = img.shape
    # for i in range(y):
      #  for j in range(x):
       #     if img[i][j] == 222 :
        #        print(1)

    return img

def Otsu_edge_thresholding(img):

    img = np.array(img,dtype = np.uint8)
    Histogram,bin_edges = np.histogram(img,bins=np.arange(257),density= True)

    Histogram = Histogram[1:]
    bin_edges = bin_edges[1:]

    Var_b_array = np.zeros((1,255))
    for k in range(255):
        with np.errstate(divide='ignore', invalid='ignore'):

            P1=Histogram[0:k]
            P2=Histogram[k:]

            C1 = Histogram[0:k]* (1+ np.arange(k))
            C2 = Histogram[k:] * (k + np.arange(255 - k))

            M1 = (1/P1.sum())*C1.sum()
            M2 = (1/P2.sum())*C2.sum()

            a=P1.sum()
            b=P2.sum()

            Var_b = a*b*(M1-M2)**2
            Var_b_array[0][k]= Var_b

    Var_b_array = np.nan_to_num(Var_b_array)
    Mg_array = Histogram * np.arange(255)
    Mg = Mg_array.sum()
    Var_g = np.var(img)

    k_opt = np.argmax(Var_b_array)
    Sep = Var_b_array[0][k_opt]/Var_g

    #img = img > k_opt
    #img = 255*(img.astype(np.int))
    #img = np.array(img, dtype=np.uint8)

    #Var_g_ex = (np.square((np.arange(256) - Mg))*Histogram).sum()
    #x = a*M1 + b*M2 - Mg
    #print(np.amin(img))
    #print(np.amax(img))
    #print(Histogram.sum())

    # y, x = img.shape
    # for i in range(y):
      #  for j in range(x):
       #     if img[i][j] == 222 :
        #        print(1)

    return k_opt

def Multiple_threshold(img):

    img = np.array(img,dtype = np.uint8)
    Histogram,bin_edges = np.histogram(img,bins=np.arange(257),density= True)
    Var_b_array = np.zeros((256,256))
    Mg = np.mean(img)

    for a in range(255):
        with np.errstate(divide='ignore', invalid='ignore'):
            P1=Histogram[0:a]
            for b in range(a,256):
                P2=Histogram[int(a):int(b)]
                P3=Histogram[int(b):]

                C1 = Histogram[0:int(a)]*np.arange(a)
                C2 = Histogram[int(a):int(b)]*(a + np.arange(b - a))
                C3 = Histogram[int(b):]*(int(b) + np.arange(256-b))

                M1 = (1/P1.sum())*C1.sum()
                M2 = (1/P2.sum())*C2.sum()
                M3 = (1/P3.sum())*C3.sum()

                x=P1.sum()
                y=P2.sum()
                z=P3.sum()

                Var_b = x*(M1-Mg)**2 + y*(M2-Mg)**2 + z*(M3-Mg)**2
                Var_b_array[int(a)][int(b)]= Var_b

    Var_b_array = np.nan_to_num(Var_b_array)
    Var_g = np.var(img)
    max = np.amax(Var_b_array)
    k_opt = np.where(Var_b_array==max)
    k_opt = np.asarray(k_opt)
    #Sep = Var_b_array[k_opt[0]][k_opt[1]]/Var_g

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] <= k_opt[0]:
                img[i][j]= 0
            elif img[i][j] > k_opt[0] and img[i][j] <= k_opt[1]:
                img[i][j]= 128
            elif img[i][j] > k_opt[1] :
                img[i][j]= 255

    img = np.array(img, dtype=np.uint8)

    #Var_g_ex = (np.square((np.arange(256) - Mg))*Histogram).sum()
    #x = a*M1 + b*M2 - Mg
    #print(np.amin(img))
    #print(np.amax(img))
    #print(Histogram.sum())

    # y, x = img.shape
    # for i in range(y):
      #  for j in range(x):
       #     if img[i][j] == 222 :
        #        print(1)

    return img


def image_partitioned_thresholding(img):

    b,a = img.shape
    block1 = img[:int(b/2),:int(a/3)]
    block2 = img[:int(b/2),int(a/3):int(2*a/3)]
    block3 = img[:int(b/2),int(2*a/3):]

    block4 = img[int(b/2):,:int(a/3)]
    block5 = img[int(b/2):,int(a/3):int(2*a/3)]
    block6 = img[int(b/2):,int(2*a/3):]

    Otsu1= Otsu_thresholding(block1)
    Otsu2= Otsu_thresholding(block2)
    Otsu3= Otsu_thresholding(block3)
    Otsu4= Otsu_thresholding(block4)
    Otsu5= Otsu_thresholding(block5)
    Otsu6= Otsu_thresholding(block6)

    result1 = np.hstack((Otsu1,Otsu2,Otsu3))
    result2 = np.hstack((Otsu4,Otsu5,Otsu6))
    result =  np.vstack((result1,result2))
    #histogram(block1)
    #histogram(block2)
    #histogram(block3)
    #histogram(block4)
    #histogram(block5)
    #histogram(block6)
    return  result

def local_threshold(img):

    y, x = img.shape
        # y = y - m + 1
        # x = x - m + 1
    std = np.zeros((y,x))
    result = np.zeros((y, x))
    Mg = np.mean(img)

    image_reflect_padding = np.pad(img, 1, 'reflect')
    for i in range(y):
        for j in range(x):
            std[i][j] = np.std(image_reflect_padding[i:i+3, j:j+3])
            if img[i][j] > 30* std[i][j] and img[i][j] > 1.5* Mg :
                result[i][j] = 255


    #max = np.amax(result)
    #result = (255/max)*result

    result = np.array(result,dtype = np.uint8)

    return result

def Moving_average(img,n,c):

    y, x= img.shape
    img_flip = np.fliplr(img)
    img_new = np.zeros((y,x))
    result = np.zeros((y,x))
    for i in range(y):
        if i%2==0 :
            img_new[i,:] = img[i,:]
        elif i%2==1 :
            img_new[i,:] = img_flip[i,:]

    img_flat = img_new.flatten()
    b = np.asarray(img_flat.shape)
    length = b[0]
    img_flat_average = np.zeros((1,length))
    moving = np.zeros((1,n))
    for k in range(length):
        with np.errstate(divide='ignore', invalid='ignore'):
            moving = img_flat[k-n:k]
            img_flat_average[0,k] = np.mean(moving)

    img_flat_average = np.nan_to_num(img_flat_average)
    img_flat_average = img_flat_average.reshape(y,x)

    for i in range(y):
        for j in range(x):
            if img[i][j] > int(c*img_flat_average[i][j]):
                result[i][j] = 255
            else:
                result[i][j] = 0

    result = np.array(result,dtype=np.uint8)

    return result

##Filter

def Spatial_convolution(img, filter):
    m, n = filter.shape
    if (m == n):
        y, x = img.shape
        #y = y - m + 1
        #x = x - m + 1
        zp = int((m - 1) / 2)
        result = np.zeros((y, x))
        image_reflect_padding = np.pad(img,zp,'reflect')
        for i in range(y):
            for j in range(x):
                result[i][j] = np.sum(image_reflect_padding[i:i+m,j:j+n] * filter)
                if result[i][j] <0:
                    result[i][j] = -result[i][j]

                  #result_zero_padding[i+zp][j+z0p] = result[i][j]

        #result = result[int((m - 1) / 2):img.shape[0] - 2, int((m - 1) / 2):img.shape[1] - 2]


        #max = np.amax(result)
        #result = (255 / max) * result

        #result = np.array(result, dtype=np.uint8)

        #result = result + img

        #result = np.array(result,dtype=float)
        #for i in range(y):
        #    for j in range(x):
        #        if result[i][j] >255:
        #            print(result[i][j])

        #result_zero_padding = img - result_zero_padding
        #b,a =result_zero_padding.shape
        #for i in range(b):
        #   for j in range(a):
        #     if result_zero_padding[i][j] <0:
        #         result_zero_padding[i][j]=0

        #result_zero_padding = np.array(result_zero_padding,dtype=np.uint8)
    return result

def S_smoothing_linear(img,c):

    smoothing_filter = np.ones((c,c))*(1/c**2)
    result = Spatial_convolution(img,smoothing_filter)
    result = np.array(result,dtype=np.uint8)

    ## result_Cut => kopt =76 , No_cut => kopt 64
    return  result

def S_Laplacian_filter(img):
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacian_diagonal = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #result1 = Spatial_convolution(img,laplacian)
    result2 = Spatial_convolution(img,laplacian_diagonal)
    max = np.amax(result2)
    result = (255 / max) * result2

    result = np.array(result,dtype=np.uint8)

    return result

def Sobel_abs(img):
    sobel_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    result_h = Spatial_convolution(img,sobel_h)
    result_h = np.array(result_h,dtype = float)
    result_v = Spatial_convolution(img,sobel_v)
    result_v = np.array(result_v,dtype = float)
    result = result_h + result_v

    max = np.amax(result)
    result = (255 / max) * result

    result = np.array(result,dtype=np.uint8)

    return result

def percentile_threshold(img,c):

    Threshold = np.percentile(img,c)
    img = img > Threshold
    img = img.astype(np.int)
    img = np.array(img,dtype=np.uint8)

    return img


main()