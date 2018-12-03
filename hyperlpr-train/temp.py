import numpy as np
from captcha.image import ImageCaptcha
import random

characters = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
print(len(characters))       

height = 168
width = 60
n_class = 36
n_len = 7

def gen(batch_size=128):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        generator = ImageCaptcha(width=width, height=height)
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(7)])
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
        yield X, y

if __name__ == "__main__":
    X_test, y_test = gen(128)
    
    print(x.shape,y.shape)