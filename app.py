import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2

model = keras.models.load_model('models/happy_sad_classification.h5')
def predict_result(result):
    if result> 0.5: 
        print(f'Predicted class is Sad')
        
    else:
        print(f'Predicted class is Happy')
        

def main():
    inp_image= input('enter image path')
    img = cv2.imread(inp_image)
    resize = tf.image.resize(img, (256,256))
    result = model.predict(np.expand_dims(resize/255, 0))
    predict_result(result)

if __name__ == '__main__':
    main()