from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import pylab 

generator = ImageCaptcha(width=128, height=40)
random_str = '1A4J'
X = generator.generate_image(random_str)
X = np.expand_dims(X, 0)

# y_pred = model.predict(X)
# plt.title('real: %s\npred:%s'%(random_str, decode(y_pred)))

plt.imshow(X[0], cmap='gray')
plt.title(random_str)

pylab.show() 