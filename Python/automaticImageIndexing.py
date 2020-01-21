class AutomaticImageIndexing:
    import cv2 as cv
    import numpy as np

    _version = '0.0'
    data_directory = "Data/"
    
    def __init__(self):
        self.data_directory = "Data/"
        
    def version(self):
        '''
        Return the version of this class
        '''
        return self._version
    
    def canny(self, image_path):
        '''
        Applys Canny algoritm to clean picture before using it to the automatic indexing
        '''
        image = self.cv.imread(image_path,0)
        edges = self.cv.Canny(image,100,200)
        return image, edges
    
    def stefen_harris(self, image_path):
        '''
        Applys Stefen Harris algoritm to detect the corner of the picture before using it to the automatic indexing
        '''
        original = self.cv.imread(image_path)
        edges = self.cv.cvtColor(original,self.cv.COLOR_BGR2GRAY)
        cannyPicture = self.cv.Canny(edges,200,400)
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
    
    def featureExtraxtion(self):
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
