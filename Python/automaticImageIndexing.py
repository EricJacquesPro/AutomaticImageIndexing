class AutomaticImageIndexing:
    import cv2 as cv
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import os.path

    #import progressbar
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.cluster import KMeans
    import sklearn.model_selection as model_selection
    #from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from xml.dom import minidom
    
    _version = '0.4'
    data_directory = "Data/"
    
    working_directory = '/'
    annotation_directory = 'annotation/'
    picture_directory = 'images/'
    picture_extension = '.jpg'
    slash = "/"
    
    n_limite_images = 0
    n_limite_directories = 0
    
    feature_generator = "SIFT"
    feature_size_min = 31
    n_feature = 200
    n_clusters = 10
    
    name_dict = {}
    desc_list = []
    kp_list = []
    picture_detail_list = []
        
        
    clf = None
    
    def __init__(self):
        self.data_directory = "Data/"
        self.working_directory = '/'
        self.annotation_directory = 'annotation/'
        self.picture_directory = 'images/'
        self.picture_extension = '.jpg'
        self.slash = "/"
        
        self.name_dict = {}
        self.desc_list = []
        self.kp_list = []
        #self.sub_picture_list = []
        self.picture_detail_list = []
        self.clf = self.SVC()
        
        self.n_clusters = 100
        self.kmeans_obj = self.KMeans(n_clusters = self.n_clusters)
        self.kmeans_ret = None#self.kmeans_obj.fit_predict(self.np.array([]))
        
    def version(self):
        '''
        Return the version of this class
        '''
        return self._version
    
    
    def read_picture_and_shift_feature_generation(self, with_sub_picture = True):
        return self.read_picture_and_feature_generation(descriptor_generator="SIFT", with_sub_picture = True)
        
    def read_picture_and_feature_generation(self, descriptor_generator="SIFT", with_sub_picture = True, version=""):
        label_count = 0

        #sift=self.cv.SIFT()
        
        if self.n_limite_directories > 0:
            folder_list = self.os.listdir(''.join([self.working_directory, self.annotation_directory]))[0:self.n_limite_directories]
        else:
            folder_list = self.os.listdir(''.join([self.working_directory, self.annotation_directory]))
                
        for folder in folder_list:
            self.name_dict[label_count] = folder
            label_count = label_count + 1
            print "Computing Features for ", folder
            anotation_directory_path = ''.join([self.working_directory,self.annotation_directory,folder])
            picture_directory_path = ''.join([self.working_directory,self.picture_directory,folder])
            if self.n_limite_images > 0:
                doc_list = self.os.listdir(anotation_directory_path)[0:self.n_limite_images]
            else:
                doc_list = self.os.listdir(anotation_directory_path)
            for doc in doc_list:
                #print(doc)
                picture_path = ''.join([picture_directory_path,self.slash,doc,self.picture_extension])
                anotation_doc = self.minidom.parse(''.join([anotation_directory_path,self.slash,doc]))
                items = anotation_doc.getElementsByTagName('object')
                #if espece pas dans name_dict, on ajoute et on incremente label_count
                if(version == "V2"):
                    temp_desc_list, temp_kp_list, temp_picture_detail_list = self.read_descriptor_picture_from_annotation_file_V2(picture_path,descriptor_generator,items,with_sub_picture)
                    self.desc_list = self.desc_list + temp_desc_list
                    self.kp_list = self.kp_list + temp_kp_list                    
                else:
                    temp_desc_list, temp_kp_list, temp_picture_detail_list = self.read_descriptor_picture_from_annotation_file(picture_path,descriptor_generator,items,with_sub_picture)
                    self.desc_list = self.desc_list + temp_desc_list
                    self.kp_list = self.kp_list + temp_kp_list
                if with_sub_picture :
                    self.picture_detail_list = self.picture_detail_list + temp_picture_detail_list
        #dictionary_size = len(self.desc_list)
    
    
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
                image, xmin, ymin, xmax, ymax = self.read_picture_from_annotation_bndbox(picture_path, bndbox)
                edges = self.canny_from_image(image)
                is_gray = True
                kp, desc = self.feature_sift_generation(edges, is_gray, False)
                row.append(kp)
                row.append(desc)
                #row.append(edges)
                row.append(dog_specie)
                images.append(row)
        return images
    
    def read_descriptor_picture_from_annotation_file(self, picture_path, descriptor_generator="SIFT", items=None, with_sub_picture=True):
        desc_list=[]
        kp_list=[]
        picture_details_list=[]
        name=''
        for element in items:
            dog_specie = element.getElementsByTagName('name')[0].firstChild.nodeValue
            bndboxs = element.getElementsByTagName('bndbox')
            for bndbox in bndboxs:
                row = []
                image, xmin, ymin, xmax, ymax = self.read_picture_from_annotation_bndbox(picture_path, bndbox)
                edges = self.canny_from_image(image)
                
                kps, descriptors = self.feature_generation(edges, is_gray = True, with_sub_picture = with_sub_picture)
                
                if((descriptors is not None) and len((descriptors)>0)):
                    #kp, desc = sift.detectAndCompute(gray,None)
                    desc_list.extend(descriptors)
                    kp_list.extend(kps)
                    #picture_details_list = []
                    if with_sub_picture :
                        for k in kps:
                            picture_details_list.extend([[picture_path, xmin, ymin, xmax, ymax]] )
                
        return desc_list, kp_list, picture_details_list
    
    def read_descriptor_picture_from_annotation_file_V2(self, picture_path, descriptor_generator="SIFT", items=None, with_sub_picture=False):
        desc_list=[]
        kp_list=[]
        picture_details_list=[]
        name=''
        for element in items:
            dog_specie = element.getElementsByTagName('name')[0].firstChild.nodeValue
            bndbox = element.getElementsByTagName('bndbox')[0]
            row = []
            image, xmin, ymin, xmax, ymax = self.read_picture_from_annotation_bndbox(picture_path, bndbox)
            edges = self.canny_from_image(image)

            kps, descriptors = self.feature_generation(edges, is_gray = True, with_sub_picture = with_sub_picture)

            if((descriptors is not None) and len((descriptors)>0)):
                desc_list.extend(descriptors)
                kp_list.extend(kps)
            else:
                kps=[]
                descriptors=[]
            picture_details_list.extend([[picture_path, xmin, ymin, xmax, ymax, kps, descriptors, dog_specie]] )
                
        return desc_list, kp_list, picture_details_list

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
        return image, xmin, ymin, xmax, ymax
    
    def read_crop_image(self, image_path, xmin, ymin, xmax, ymax):#    def read_crop_image(self, image_path, xmin, xmax, ymin, ymax):
        '''
        read image with open cv
        '''
        #print('...')
        if self.os.path.exists(image_path):
            image = self.cv.imread(image_path)
            #print(image)
            xmin=int(xmin)
            ymin=int(ymin) 
            xmax=int(xmax)
            ymax=int(ymax)
            image = image[ymin: ymax, xmin: xmax].copy()
            #image = self.crop_image(image, xmin, ymin, xmax, ymax)
            #print(image)
            return image
        
        return None
    
    def crop_image(self, image, xmin, ymin, xmax, ymax):
        '''
        crop image with open cv
        '''
        xmin=int(xmin)
        ymin=int(ymin) 
        xmax=int(xmax)
        ymax=int(ymax)

        image = image[ymin: ymax, xmin: xmax].copy()
        return image
        
    def read_image(self, image_path, crop):
        '''
        read image with open cv
        '''
        image = self.cv.imread(image_path)
        if crop :
            image = image[1:180,71:192].copy()
        return image
    
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

        sift=self.cv.SIFT(nfeatures=self.n_feature)
        kp = sift.detect(gray,None)
        image = self.cv.drawKeypoints(gray,kp,image)
        image_sifted = self.cv.drawKeypoints(image,kp,color=(0,255,0), flags=0)
        return image, image_sifted
    
    def feature_generation(self, image, is_gray = True, with_sub_picture=False):
        if self.feature_generator == "ORB":
            generator = self.cv.ORB(nfeatures=self.n_feature, patchSize=self.feature_size_min)
        else : 
            generator = self.cv.SIFT(nfeatures=self.n_feature)
            
        if is_gray :
            gray = image
        else :
            gray= self.cv.cvtColor(image,self.cv.COLOR_BGR2GRAY)
            
        kp, desc = generator.detectAndCompute(gray,None)
        return kp, desc 
    
    def feature_sift_generation(self, image, is_gray = True, with_sub_picture=False):
        sift=self.cv.SIFT(nfeatures=self.n_feature)
        if is_gray :
            gray = image
        else :
            gray= self.cv.cvtColor(image,self.cv.COLOR_BGR2GRAY)
            
        kp, desc = sift.detectAndCompute(gray,None)   
        return kp, desc 
    
    def feature_orb_generation(self, image, is_gray = True, with_sub_picture=False):
        orb=self.cv.ORB(nfeatures=self.n_feature)
        
        if is_gray :
            gray = image
        else :
            gray= self.cv.cvtColor(image,self.cv.COLOR_BGR2GRAY)

        kp, desc = orb.detectAndCompute(gray,None)
        
        return kp, desc
    
    def kp_to_feature(self, image, kp):
        """Convert KeyPoint to Feature."""
        x, y = kp.pt
        radius = kp.size / 2
        weight = radius * kp.response ** 2
        '''
            weight=weight,
            xmin=x - radius,
            ymin=y - radius,
            xmax=x + radius,
            ymax=y + radius
         #* self._padding
         '''
        print(radius)
        return self.crop_image(image, int(x-radius), int(y-radius), int(x+radius), int(y+radius))
    
    def kp_to_picture(self, num_kp):
        #print(num_kp)
        #print(self.picture_detail_list[num_kp])
        kp = self.kp_list[num_kp]
        picture_detail = self.picture_detail_list[num_kp]
        x, y = kp.pt
        radius = kp.size / 2
        weight = radius * kp.response ** 2
        '''
            weight=weight,
            xmin=x - radius,
            ymin=y - radius,
            xmax=x + radius,
            ymax=y + radius
         #* self._padding
         '''
        image_temp = self.read_crop_image(picture_detail[0], picture_detail[1], picture_detail[2], picture_detail[3], picture_detail[4])
        image_temp = self.canny_from_image(image_temp)
        return self.crop_image(image_temp, int(x-radius), int(y-radius), int(x+radius), int(y+radius))
    
    def feature_sift_showing(self, image, kp):
        return self.plt.imshow(self.cv.drawKeypoints(image, kp, color_img.copy()))

    
    def feature_sift(self, image1, image2, is_gray):
        octo_front_kp, octo_front_desc = self.feature_sift_generation(image1, is_gray, False)
        octo_offset_kp, octo_offset_desc = self.feature_sift_generation(image2, is_gray, False)
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
        image1_kp, image1_desc = self.feature_sift_generation(image1, is_gray, False)
        image2_kp, image2_desc = self.feature_sift_generation(image2, is_gray, False)
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

    def clusturing(self, n_clusters = None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        self.kmeans_obj = self.KMeans(n_clusters = self.n_clusters)
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.np.array(self.desc_list))

    def developVocabulary(self, n_images):
        return developBagOfVisualWord()
    
    def developBagOfVisualWord(self):
        n_images = len(self.name_dict)
        mega_histogram = self.np.array([self.np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(self.desc_list[i])
            for j in range(l):
                idx = self.kmeans_ret[old_count+j]
                mega_histogram[i][idx] += 1
            old_count += l
        print "Vocabulary Histogram Generated"
        return mega_histogram
    #representer 1 cluster pour voir l'homogeneite?
    #=>representation des descriptor du cluster

    def plotHist(self, mega_histogram, n_clusters, vocabulary = None):
        print "Plotting histogram"
        if vocabulary is None:
            vocabulary = mega_histogram

        x_scalar = self.np.arange(n_clusters)
        y_scalar = self.np.array([abs(self.np.sum(vocabulary[:,h], dtype=self.np.int32)) for h in range(self.n_clusters)])

        self.plt.bar(x_scalar, y_scalar)
        self.plt.xlabel("Visual Word Index")
        self.plt.ylabel("Frequency")
        self.plt.title("Vocabulary Generated")
        self.plt.xticks(x_scalar + 0.4, x_scalar)
        self.plt.show()
    
    def plot_descriptor_in_cluster_subplot(self, num_cluster, number_column = 10):
        number_descriptor = self.np.count_nonzero(self.kmeans_ret == num_cluster)
        if number_descriptor > 0:
            number_row = int(number_descriptor / number_column)+1
            height = number_row*12
            width = number_column*8
            
            #if height > 84672:
            print height
            fig,axes = self.plt.subplots(
                nrows = number_row,
                ncols = number_column, 
                figsize=(width,height),
                subplot_kw={'xticks': [], 'yticks': []}
            )
            current_cloumn = 0
            current_row = 0
            list_item = range(len(self.desc_list))
            for number_item in list_item:
                current_descriptor_cluster = self.kmeans_ret[number_item]
                if current_descriptor_cluster == num_cluster:
                    image = self.kp_to_picture(number_item)    
                    if current_cloumn == number_column:
                        current_row = current_row+1
                        current_cloumn = 0
                    axes[current_row,current_cloumn].imshow(image,
                                                            interpolation=None, 
                                                            cmap='viridis'
                                                           )
                    current_cloumn = current_cloumn+1
            #self.plt.title("Decriptor near the cluster")
            self.plt.show()
        else:
            print("No picture for this cluster. Try an other cluster")
    
    def plot_descriptor_in_cluster_subplot_V2(self, num_cluster, number_column = 10):
        number_descriptor = self.np.count_nonzero(self.kmeans_ret == num_cluster)
        if number_descriptor > 0:
            number_row = int(number_descriptor / number_column)+1
            height = number_row*12
            width = number_column*8
            
            #if height > 84672:
            print "-------"
            print height
            print width
            print "-------"
            fig,axes = self.plt.subplots(
                ncols = number_column, 
                figsize=(width,12),
                subplot_kw={'xticks': [], 'yticks': []}
            )
            current_cloumn = 0
            current_row = 0
            list_item = range(len(self.desc_list))
            for number_item in list_item:
                current_descriptor_cluster = self.kmeans_ret[number_item]
                if current_descriptor_cluster == num_cluster:
                    image = self.kp_to_picture(number_item)    
                    if current_cloumn == number_column:
                        self.plt.show()
                        fig,axes = self.plt.subplots(
                            ncols = number_column, 
                            figsize=(width,12),
                            subplot_kw={'xticks': [], 'yticks': []}
                        )
                        current_cloumn = 0
                    axes[current_cloumn].imshow(image,
                                                  interpolation=None, 
                                                  cmap='viridis'
                                                 )
                    current_cloumn = current_cloumn+1
            #self.plt.title("Decriptor near the cluster")
            self.plt.show()
        else:
            print("No picture for this cluster. Try an other cluster")
    
    def plot_descriptor_in_cluster_individualplot(self, num_cluster):
        number_descriptor = self.np.count_nonzero(self.kmeans_ret == num_cluster)
        if number_descriptor > 0:
            list_item = range(len(self.desc_list))
            for number_item in list_item:
                current_descriptor_cluster = self.kmeans_ret[number_item]
                if current_descriptor_cluster == num_cluster:
                    image = self.kp_to_picture(number_item)    
                    self.plt.figure(figsize=(12,6))
                    self.plt.imshow(image)
                    self.plt.show()
        else:
            print("No picture for this cluster. Try an other cluster")
    
    def plot_descriptor_in_cluster(self, num_cluster):
        nombre_desc_cluster = len([ num for num in self.kmeans_ret if num == 1 ])
        nombre_colonne = 10
        nombre_ligne = 1+ int(nombre_desc_cluster / nombre_colonne)+1
        fig,axes = self.plt.subplots(nrows = nombre_ligne, ncols = nombre_colonne, figsize=(80,300),
                                                      subplot_kw={'xticks': [], 'yticks': []})

        a=0
        b=0

        for i in range(len(self.desc_list)):
            if self.kmeans_ret[i] == num_cluster:
                if b == nombre_colonne:
                    a = a + 1
                    b = 0
                axes[a,b].imshow(self.desc_list[i].reshape(16,8), interpolation=None, cmap='viridis')
                b=b+1
        self.plt.show()
    

    def plot_picture_from_cluster(self, num_cluster):
        n_images = len(self.name_dict)
        mega_histogram = self.np.array([self.np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(self.desc_list[i])
            for j in range(l):
                idx = self.kmeans_ret[old_count+j]
                if idx == num_cluster :
                    self.plt.subplot(111)
                    self.plt.imshow(image,cmap = 'gray')
                    break
        self.plt.title("Decriptor near the cluster")
        self.plt.show()

    def standardize(self, mega_histogram, std=None):
        """
        standardize is required to normalize the distribution
        """
        if std is None:
            scale = self.StandardScaler().fit(mega_histogram)
            mega_histogram = scale.transform(mega_histogram)
        else:
            print "STD not none. External STD supplied"
            mega_histogram = std.transform(mega_histogram)
        return mega_histogram

    def picture_in_BOV (self, picture_path, picture_descriptors, show_graph=None, reload_descriptor=None):
        if (reload_descriptor is None) or (reload_descriptor == False):
            image_descriptions = picture_descriptors
        else:
            image, image_filtered = self.canny(image_path=picture_path)
            is_gray = True
            image_kp, image_descriptions, image_partials = self.feature_sift_generation(image_filtered, is_gray, False)

        image_clustered = self.kmeans_obj.predict(image_descriptions)
        # generate vocab for test image
        clusters = self.np.array( [[ 0 for i in range(self.n_clusters)]])
        # locate nearest clusters for each of 
        # the visual word (feature) present in the 0:6
        # print vocab
        for each in image_clustered:
            clusters[0][each] += 1
        
        if (show_graph is None) or (show_graph == True):    
            self.plotHist(clusters, self.n_clusters, None)

        return clusters
    
    def train_predict(self, with_reload_descriptor=None, with_performance_measure=None):
        X=[k[6] for k in self.picture_detail_list] # kps save previously
        X_desc = []
        for x_temp in X:
            if (with_reload_descriptor is None) or (with_reload_descriptor == False):
                X_desc.extend(self.picture_in_BOV(picture_path=None, picture_descriptors=x_temp, show_graph=False))
            else:
                X_desc.extend(self.picture_in_BOV(picture_path=x_temp[0], picture_descriptors=None, show_graph=False))
        del X
        
        y=[r[7] for r in self.picture_detail_list] # result save previously
        X_train, X_test, y_train, y_test = self.model_selection.train_test_split(X_desc, y, test_size=0.25, random_state=42)
        #X_train, y_train = X_desc, y
        """
        uses sklearn.svm.SVC classifier (SVM) 
        """
        self.clf.fit(X_train, y_train)
        
        if (with_performance_measure is None) or (with_performance_measure == True):
            score = self.clf.score(X_train, y_train )#X_test, y_test)
            print"predict score = %s" % score
        
        print "Training completed"
    
    def train(self, mega_histogram, train_labels):
        """
        uses sklearn.svm.SVC classifier (SVM) 
        """
        self.clf.fit(mega_histogram, train_labels)
        print "Training completed"

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
