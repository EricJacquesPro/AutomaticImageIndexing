class NeuronalImageIndexing:
    '''
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.vgg16 import VGG16
    '''
    from PIL import Image
    
    import sys
    from PIL import Image
    sys.modules['Image'] = Image 

    from PIL import Image
    print(Image.__file__)
    import Image
    print(Image.__file__)
    
    version="0"
    
    def __init__(self):
        self.version = "1"
        
    def transfert_learning_vgg16(
        image_folder_path=None,#r'C:\Users\naru_\OneDrive\Documents\openclassroom\P7 traitement image\imagesClassees',
        mode=None, 
        size_batch=32,
        nb_epoch=10,
        lr=0.001
    ):
        from keras.layers import Input, Lambda, Dense, Flatten
        from keras.models import Model
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
        from keras.preprocessing.image import load_img, img_to_array
        from keras.preprocessing import image
        from keras.layers import Input, Flatten, Dense
        from keras import optimizers
        from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
        import numpy as np
        #import cv2 as cv
        from PIL import Image

        #from sklearn.metrics import confusion_matrix
        #import matplotlib.pyplot as plt
        #%matplotlib inline
        from glob import glob

        import sys
        from PIL import Image
        sys.modules['Image'] = Image 

        batch_size = size_batch
        epochs = nb_epoch
        verbose=5

        
        gen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            preprocessing_function=preprocess_input,
        )
        # get label mapping for confusion matrix plot later
        test_gen = gen.flow_from_directory(
            image_folder_path,#'C:/Users/naru_/OneDrive/Documents/openclassroom/P7 traitement image/imagesClassees', 
            target_size=(224,224),
            color_mode="rgb",
            classes=None,
            class_mode="categorical",
            batch_size=batch_size,
            save_format="jpg",
            subset='training'
        )
        classes = test_gen.class_indices
        print("classes:")
        print(classes)

        val_generator = gen.flow_from_directory(
            image_folder_path,#'C:/Users/naru_/OneDrive/Documents/openclassroom/P7 traitement image/imagesClassees', 
                    target_size=(224,224),
            color_mode="rgb",
            classes=None,
            class_mode="categorical",
            batch_size=batch_size,
            save_format="jpg",
            subset='validation'
        )
        
        nb_classes=len(classes) 
        # Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        #model = VGG16()
        for i, layer in enumerate(model.layers):
            print (i, layer.name, layer.output_shape)
        print(model.summary())

        # Récupérer la sortie de ce réseau
        x = model.output

        # Ajouter la nouvelle couche fully-connected pour la classification à 120 classes
        predictions = Dense(nb_classes, activation='softmax')(x)

        # Définir le nouveau modèle
        new_model = Model(inputs=model.input, outputs=predictions)

        if(mode==1 or mode=='fine-tuning total'):
            for layer in model.layers:
                layer.trainable = True
        if (mode==2 or mode=='extraction de features'):
            for layer in model.layers:
                layer.trainable = False
        if (mode==3 or mode=='fine-tuning partiel'):
            # Ne pas entraîner les 10 premières couches (les plus basses) 
            for layer in model.layers[:10]:
                layer.trainable = False

        # Compiler le modèle 
        new_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False
        ), metrics=["accuracy"])

        # create an instance of ImageDataGenerator
        '''
        gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input,
            validation_split=0.2
        )
        '''
        # Entraîner sur les données d'entraînement (X_train, y_train)
        #print(test_gen[1].shape)
        model_info = new_model.fit(test_gen,
                                   #y_train, 
                                   epochs=epochs,
                                   validation_data=val_generator,
                                   verbose=1
        )
        return new_model, model_info, classes

    def predict(
        image_path,
        model,
        classes, 
        number_top_predict
    ):
        from keras.preprocessing.image import load_img, img_to_array
        from keras.applications.vgg16 import decode_predictions
        from keras.applications.vgg16 import preprocess_input
        from keras.applications.vgg16 import VGG16

        from PIL import Image
        import numpy as np

        img = load_img(image_path)
        img = img.resize((224, 224))

        img = img_to_array(img)  # Convertir en tableau numpy
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
        img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

        y_after = model.predict(img)  # Prédir la classe de l'image (parmi les 1000 classes d'ImageNet)

        print(y_after)

        key_list = list(classes.keys()) 
        val_list = list(classes.values()) 


        better = np.where(y_after[0] == np.amax(y_after[0]))
        print('better element from Numpy Array : ', better[0][0])
        #print(especes_partial)
        print('surement un ', key_list[val_list.index(better[0][0])])

        # Afficher les 3 classes les plus probables
        #print('Top 3 :', decode_predictions(y_after, top=3)[0])
        #return decode_predictions(y_after, top=3)[0]
        return y_after
    #result = predict(modelTF, classes, number_predict)
    #print('Top ',number_predict, ' :', predict(modelTF, number_predict))
    
    def scoring( historique):
        import matplotlib.pyplot as plt
        plt.subplots_adjust(hspace=0.4)
        plt.subplot(131)
        plt.plot(historique.history["accuracy"])
        plt.plot(historique.history['val_accuracy'])
        plt.plot(historique.history['loss'])
        plt.plot(historique.history['val_loss'])
        plt.subplot(132)
        plt.plot(historique.history["accuracy"])
        plt.plot(historique.history['val_accuracy'])
        plt.subplot(133)
        plt.plot(historique.history['loss'])
        plt.plot(historique.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy/loss")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
        plt.show()
        
    def scoringCompare(array_of_historique, legende):
        import matplotlib.pyplot as plt
        for historique in array_of_historique :
            plt.plot(historique.history["accuracy"])
            plt.plot(historique.history['val_accuracy'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(legende)
        plt.show()
