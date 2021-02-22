
# DIP_Morphology & Thresholding
## This repository is python code of "Digital Image Processing, Rafael C.Gonzalez, 3rd edition"
### You can see more details in [PDF](https://github.com/SeongSuKim95/DIP_Morphology_and_Thresholding/blob/master/Thresholidng_and_Morphology.pdf)

* Thresholding
  * Histogram
    ```python
    def histogram(img):
        plt.xlabel("Intensity")
        plt.ylabel("Pixel Frequency")
        plt.title("Histogram")
        plt.hist(img.ravel(),255,[1,256],density=True)
        plt.show()
        return
    ```
  * Effect of noise ( histogram )
    ![Effect_of_noise](https://user-images.githubusercontent.com/62092317/108644397-1baa8600-74f2-11eb-9950-bc8ca66f6156.PNG)
  * Basic Global Thresholding
    ![Basic_Global_Thresholding](https://user-images.githubusercontent.com/62092317/108644602-13067f80-74f3-11eb-891d-23a59a99dc7b.PNG)
     ```python
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
     ```     
  * Otsu Thresholding
    ![Otsu_thresholding](https://user-images.githubusercontent.com/62092317/108644393-1a795900-74f2-11eb-9a49-52e7f7358ca1.png)
    ```python
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

        return img
    ```
  
  * Improvement of Otsu 
    * Using Smoothing (histogram)
      * Before Smoothing
      ![Otsu_after_smoothing](https://user-images.githubusercontent.com/62092317/108644392-19e0c280-74f2-11eb-973d-884ce753fcc0.PNG)
      * After Smoothing
    
    * Using Edges (Sobel) (See details in pdf)
    
    * Using Edges (Laplacian) (See details in pdf)
      ![Thresholding_using_laplacian](https://user-images.githubusercontent.com/62092317/108644394-1a795900-74f2-11eb-8f7e-bf1d0b336fb7.PNG)
      ```python
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

          return k_opt
      ```

  * Multiple Thresholding
    ![Multiple_Thresholding](https://user-images.githubusercontent.com/62092317/108645736-261b4e80-74f7-11eb-9f9f-90cfb5abaf31.PNG)
    ```python
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

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] <= k_opt[0]:
                    img[i][j]= 0
                elif img[i][j] > k_opt[0] and img[i][j] <= k_opt[1]:
                    img[i][j]= 128
                elif img[i][j] > k_opt[1] :
                    img[i][j]= 255

        img = np.array(img, dtype=np.uint8)


        return img
    ```    
  * Image partitioning (Used Otsu function locally)
    ![Image_partitioning](https://user-images.githubusercontent.com/62092317/108644400-1c431c80-74f2-11eb-92c0-ce59ff500281.PNG)
    ```python
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

        return  result
    ``` 

  * Variable Thresholding
    ![Variable_Thresholding](https://user-images.githubusercontent.com/62092317/108644395-1b11ef80-74f2-11eb-947d-61efad941ab7.PNG)
    ```python
    def local_threshold(img):

        y, x = img.shape

        std = np.zeros((y,x))
        result = np.zeros((y, x))
        Mg = np.mean(img)

        image_reflect_padding = np.pad(img, 1, 'reflect')
        for i in range(y):
            for j in range(x):
                std[i][j] = np.std(image_reflect_padding[i:i+3, j:j+3])
                if img[i][j] > 30* std[i][j] and img[i][j] > 1.5* Mg :
                    result[i][j] = 255

        result = np.array(result,dtype = np.uint8)

        return result
    ```

  * Moving Average Thresholding
    ![Moving_average_Thresholding](https://user-images.githubusercontent.com/62092317/108644385-177e6880-74f2-11eb-9aab-0543f671228d.PNG)
    ```python 
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
    ```
* Morphology
   * Erosion
     ![Erosion](https://user-images.githubusercontent.com/62092317/108644398-1c431c80-74f2-11eb-933d-afd27f2afef3.png)
     ```python
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

         return result 
     ```
   
  * Dilation
   ![Dilation](https://user-images.githubusercontent.com/62092317/108644396-1baa8600-74f2-11eb-8be9-900a61b5c53d.PNG)
   ```python
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

      return result
    ```
   
  * Opening
   ![Opening](https://user-images.githubusercontent.com/62092317/108644388-19482c00-74f2-11eb-9259-40f7bca4e72d.PNG)
  * Opening and Closing
   ![Opening_and_Closing](https://user-images.githubusercontent.com/62092317/108644390-19482c00-74f2-11eb-8465-d9a8b2c8707b.PNG)
  
