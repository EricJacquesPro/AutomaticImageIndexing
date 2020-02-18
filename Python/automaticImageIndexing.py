class AutomaticImageIndexing:
    import cv2 as cv
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import os.path

    _version = '0.0'
    data_directory = "Data/"
    
    def __init__(self):
        self.data_directory = "Data/"
        
    def version(self):
        '''
        Return the version of this class
        '''
        return self._version
    
    def read_picture_from_annotation_file(self, picture_path, items):
        # total amount of items
        #bndbox = items[0].getElementsByTagName('bndbox')
        images=[]
        name=''
        for element in items:
            dog_specie = element.getElementsByTagName('name')[0].firstChild.nodeValue
            bndboxs = element.getElementsByTagName('bndbox')
            for bndbox in bndboxs:
                row = []
                image = self.read_picture_from_annotation_bndbox(picture_path, bndbox)
                edges = self.canny_from_image(image)

                print('apres canny')
                is_gray = True
                kp, desc = self.feature_sift_generation(image, is_gray)
                row.append(kp)
                row.append(desc)
                #row.append(edges)
                row.append(dog_specie)
                images.append(row)
        return images
    
    def read_descriptor_picture_from_annotation_file(self, picture_path, items):
        desc_list=[]
        name=''
        for element in items:
            dog_specie = element.getElementsByTagName('name')[0].firstChild.nodeValue
            bndboxs = element.getElementsByTagName('bndbox')
            for bndbox in bndboxs:
                row = []
                image = self.read_picture_from_annotation_bndbox(picture_path, bndbox)
                edges = self.canny_from_image(image)

                print('apres canny')
                is_gray = True
                kp, desc = self.feature_sift_generation(image, is_gray)
                #kp, desc = sift.detectAndCompute(gray,None)
                desc_list.extend(desc)
        return desc_list

    def read_picture_from_annotation_bndbox(self, picture_path, bndbox):
        xmin = bndbox.getElementsByTagName('xmin')[0].firstChild.nodeValue
        ymin = bndbox.getElementsByTagName('ymin')[0].firstChild.nodeValue
        xmax = bndbox.getElementsByTagName('xmax')[0].firstChild.nodeValue
        ymax = bndbox.getElementsByTagName('ymax')[0].firstChild.nodeValue

        image = self.read_crop_image(
            image_path=picture_path,
            xmin=int(xmin),
            ymin=int(ymin), 
            xmax=int(xmax), 
            ymax=int(ymax)
        )
        return image

    def read_crop_image(self, image_path, xmin, xmax, ymin, ymax):
        '''
        read image with open cv
        '''
        if self.os.path.exists(image_path):
            image = self.cv.imread(image_path)
            image = image[ymin: ymax, xmin: xmax].copy()
            return image
        print('no image found')
        return None;

    def read_image(self, image_path, crop):
        '''
        read image with open cv
        '''
        image = self.cv.imread(image_path)
        if crop :
            image = image[1:180,71:192].copy()
        return image;
    
    def canny(self, image_path):
        '''
        Applys Canny algoritm to clean picture before using it to the automatic indexing
        '''
        image = self.read_image(image_path, False)
        edges = self.canny_from_image(image)
        return image, edges
    
    def canny_from_image(self, image):
        '''
        Applys Canny algoritm to clean picture before using it to the automatic indexing
        '''
        edges = self.cv.Canny(image,100,200)
        return edges
    
    def stefen_harris(self, image_path, canny):
        '''
        Applys Stefen Harris algoritm to detect the corner of the picture before using it to the automatic indexing
        '''
        original = self.cv.imread(image_path)
        gray = self.cv.cvtColor(original,self.cv.COLOR_BGR2GRAY)
        if canny :
            cannyPicture = self.cv.Canny(gray,200,400)
        else:
            cannyPicture = gray
        img = self.np.float32(cannyPicture)
        corners = self.cv.cornerHarris(img,2,3,1)

        image_cornered = original
        corners2 = self.cv.dilate(corners, None, iterations=2)
        image_cornered[corners2>0.02*corners2.max()] = [255,0,0]
        return original, image_cornered, corners, corners2
    
    def preprocessing(self):
        '''
        Function to prepare picture before using it to the automatic indexing
        '''
        preprocess = 'to do'
        return preprocess
    
    def feature_sift_extration(self, image, is_gray):
        if is_gray :
            gray = image
        else :
            gray= self.cv.cvtColor(image,self.cv.COLOR_BGR2GRAY)

        sift=self.cv.SIFT()
        kp = sift.detect(gray,None)
        image = self.cv.drawKeypoints(gray,kp,image)
        image_sifted = self.cv.drawKeypoints(image,kp,color=(0,255,0), flags=0)
        return image, image_sifted
    
    def feature_sift_generation(self, image, is_gray):
        if is_gray :
            gray = image
        else :
            gray= self.cv.cvtColor(image,self.cv.COLOR_BGR2GRAY)

        sift=self.cv.SIFT()
        kp, desc = sift.detectAndCompute(gray,None)
        return kp, desc
    
    def feature_sift_showing(self, image, kp):
        return self.plt.imshow(self.cv.drawKeypoints(image, kp, color_img.copy()))

    
    def feature_sift(self, image1, image2, is_gray):
        octo_front_kp, octo_front_desc = self.feature_sift_generation(image1, is_gray)
        octo_offset_kp, octo_offset_desc = self.feature_sift_generation(image2, is_gray)
        bf = self.cv.BFMatcher(self.cv.NORM_L2, crossCheck=True)
        matches = bf.match(octo_front_desc, octo_offset_desc)
        # Sort the matches in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # draw the top N matches
        N_MATCHES = 100

        match_img = self.cv.drawMatches(
            octo_front, octo_front_kp,
            octo_offset, octo_offset_kp,
            matches[:N_MATCHES], outImg=None, flags=2)
        '''
        cv2.drawMatches(
            octo_front, octo_front_kp,
            octo_offset, octo_offset_kp,
            matches1to2[, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]])
        '''
        return match_img
    
    def feature_sift_knn(self, image1, image2, is_gray):
        image1_kp, image1_desc = self.feature_sift_generation(image1, is_gray)
        image2_kp, image2_desc = self.feature_sift_generation(image2, is_gray)
        # Create matcher
        #bf = self.cv.BFMatcher(self.cv.NORM_L2, crossCheck=True)
        
        bf = self.cv.BFMatcher(normType=self.cv.NORM_L2, crossCheck=False)
        # Perform KNN matching
        #matches = bf.knnMatch(image1_desc, image2_desc, k=2)
        matches = bf.knnMatch(self.np.asarray(image1_desc, self.np.float32), self.np.asarray(image2_desc, self.np.float32), k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
               # Add first matched keypoint to list
               # if ratio test passes
               good.append(m)

        # Or do a list comprehension
        #good = [m for (m,n) in matches if m.distance < 0.75*n.distance]

        # Now perform drawMatches
        out = self.drawMatches(image1, image1_kp, image2, image2_kp, good)
        
        
    def drawMatches(self, img1, kp1, img2, kp2, matches):
        """
        My own implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0

        This function takes in two images with their associated 
        keypoints, as well as a list of DMatch data structure (matches) 
        that contains which keypoints matched in which images.

        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.

        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.

        img1,img2 - Grayscale images
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
        """

        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        # Create the output image
        # The rows of the output are the largest between the two images
        # and the columns are simply the sum of the two together
        # The intent is to make this a colour image, so make this 3 channels
        out = self.np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

        # Place the first image to the left
        out[:rows1,:cols1] = self.np.dstack([img1, img1, img1])

        # Place the next image to the right of it
        out[:rows2,cols1:] = self.np.dstack([img2, img2, img2])

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            self.cv.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            self.cv.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            self.cv.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


        # Show the image
        self.cv.imshow('Matched Features', out)
        self.cv.waitKey(0)
        self.cv.destroyWindow('Matched Features')

        # Also return the image if you'd like a copy
        return out

    def feature_processing(self):
        '''
        Function to extract feature of picture before using it to the automatic indexing
        '''
        features = 'to do'
        return features
    
    def classification(self):
        '''
        Function to classify picture 
        '''
        features = 'to do'
        return features
