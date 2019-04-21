import torch
torch.__version__

import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline

from torch.autograd import Variable
from PIL import Image


y = torch.rand(55)

y.size()

print (y.size())


car = np.array(Image.open('car.jpg').resize( (224,224)))
car_tensor = torch.from_numpy(car)
car_tensor.size()

# Display car image
plt.imshow(car)