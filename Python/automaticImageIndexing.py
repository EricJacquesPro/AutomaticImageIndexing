class AutomaticImageIndexing:
    _version = '0.0'
    data_directory = "Data/"
    
    def __init__(self):
        self.data_directory = "Data/"
        
    def version(self):
        '''
        Return the version of this class
        '''
        return self._version
    
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
