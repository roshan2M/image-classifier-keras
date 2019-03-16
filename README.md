# image-classifier-keras
My first attempt at building an image classifier in Keras. I did this in a few steps:
* Pulled images from a file directory on computer
* Resized them into (200px x 200px) sized images
* Flattened into 40,000 dimensional vectors to feed into classifer
* Specified number of classes, epochs, filters and layers in Keras
* Split training & testing sets
* Used Dense layers to connect layers of neural networks, use ReLu intermediate activation function, and then used MaxPooling2D
* Evaluated model with testing set

## Testing the Classifier
(Trying to find out how to load ```buoy_classifier.h5``` onto GitHub since it exceeds the 100MB limit.)

Make sure the dependencies of Keras 1.2.2 are installed. Once the file ```buoy_classifier.h5``` is on GitHub, load it into a directory. Then load model using ```model = load_model('buoy_classifier.h5')``` and test it on a new image using ```model.predict(buoy, batch_size=1, verbose=1)```.
