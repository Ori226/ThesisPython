from PIL import Image
import numpy

def ImageToTheanoTensor(image_path):
    
    img = Image.open(image_path)
    # dimensions are (height, width, channel)
    img = numpy.asarray(img, dtype='float32') / 256.


    # put image in 4D tensor of shape (1, 3, height, width)
    img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
    return img_




#ImageToTheanoTensor('3wolfmoon.jpg')