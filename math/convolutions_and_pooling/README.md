
## Convolutions and Pooling

### Description
0. Valid ConvolutionmandatoryWrite a functiondef convolve_grayscale_valid(images, kernel):that performs a valid convolution on grayscale images:imagesis anumpy.ndarraywith shape(m, h, w)containing multiple grayscale imagesmis the number of imageshis the height in pixels of the imageswis the width in pixels of the imageskernelis anumpy.ndarraywith shape(kh, kw)containing the kernel for the convolutionkhis the height of the kernelkwis the width of the kernelYou are only allowed to use twoforloops; any other loops of any kind are not allowedReturns: anumpy.ndarraycontaining the convolved imagesubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 0-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':

    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ ./0-main.py 
(50000, 28, 28)
(50000, 26, 26)Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/convolutions_and_poolingFile:0-convolve_grayscale_valid.pyHelp×Students who are done with "0. Valid Convolution"Review your work×Correction of "0. Valid Convolution"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

1. Same ConvolutionmandatoryWrite a functiondef convolve_grayscale_same(images, kernel):that performs a same convolution on grayscale images:imagesis anumpy.ndarraywith shape(m, h, w)containing multiple grayscale imagesmis the number of imageshis the height in pixels of the imageswis the width in pixels of the imageskernelis anumpy.ndarraywith shape(kh, kw)containing the kernel for the convolutionkhis the height of the kernelkwis the width of the kernelif necessary, the image should be padded with 0’sYou are only allowed to use twoforloops; any other loops of any kind are not allowedReturns: anumpy.ndarraycontaining the convolved imagesubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 1-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same


if __name__ == '__main__':

    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ ./1-main.py 
(50000, 28, 28)
(50000, 28, 28)Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/convolutions_and_poolingFile:1-convolve_grayscale_same.pyHelp×Students who are done with "1. Same Convolution"Review your work×Correction of "1. Same Convolution"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

2. Convolution with PaddingmandatoryWrite a functiondef convolve_grayscale_padding(images, kernel, padding):that performs a convolution on grayscale images with custom padding:imagesis anumpy.ndarraywith shape(m, h, w)containing multiple grayscale imagesmis the number of imageshis the height in pixels of the imageswis the width in pixels of the imageskernelis anumpy.ndarraywith shape(kh, kw)containing the kernel for the convolutionkhis the height of the kernelkwis the width of the kernelpaddingis a tuple of(ph, pw)phis the padding for the height of the imagepwis the padding for the width of the imagethe image should be padded with 0’sYou are only allowed to use twoforloops; any other loops of any kind are not allowedReturns: anumpy.ndarraycontaining the convolved imagesubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 2-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding


if __name__ == '__main__':

    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ ./2-main.py 
(50000, 28, 28)
(50000, 30, 34)Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/convolutions_and_poolingFile:2-convolve_grayscale_padding.pyHelp×Students who are done with "2. Convolution with Padding"Review your work×Correction of "2. Convolution with Padding"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

3. Strided ConvolutionmandatoryWrite a functiondef convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):that performs a convolution on grayscale images:imagesis anumpy.ndarraywith shape(m, h, w)containing multiple grayscale imagesmis the number of imageshis the height in pixels of the imageswis the width in pixels of the imageskernelis anumpy.ndarraywith shape(kh, kw)containing the kernel for the convolutionkhis the height of the kernelkwis the width of the kernelpaddingis either a tuple of(ph, pw), ‘same’, or ‘valid’if ‘same’, performs a same convolutionif ‘valid’, performs a valid convolutionif a tuple:phis the padding for the height of the imagepwis the padding for the width of the imagethe image should be padded with 0’sstrideis a tuple of(sh, sw)shis the stride for the height of the imageswis the stride for the width of the imageYou are only allowed to use twoforloops; any other loops of any kind are not allowedHint: loop overiandjReturns: anumpy.ndarraycontaining the convolved imagesubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 3-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale


if __name__ == '__main__':

    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ ./3-main.py 
(50000, 28, 28)
(50000, 13, 13)Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/convolutions_and_poolingFile:3-convolve_grayscale.pyHelp×Students who are done with "3. Strided Convolution"Review your work×Correction of "3. Strided Convolution"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

4. Convolution with ChannelsmandatoryWrite a functiondef convolve_channels(images, kernel, padding='same', stride=(1, 1)):that performs a convolution on images with channels:imagesis anumpy.ndarraywith shape(m, h, w, c)containing multiple imagesmis the number of imageshis the height in pixels of the imageswis the width in pixels of the imagescis the number of channels in the imagekernelis anumpy.ndarraywith shape(kh, kw, c)containing the kernel for the convolutionkhis the height of the kernelkwis the width of the kernelpaddingis either a tuple of(ph, pw), ‘same’, or ‘valid’if ‘same’, performs a same convolutionif ‘valid’, performs a valid convolutionif a tuple:phis the padding for the height of the imagepwis the padding for the width of the imagethe image should be padded with 0’sstrideis a tuple of(sh, sw)shis the stride for the height of the imageswis the stride for the width of the imageYou are only allowed to use twoforloops; any other loops of any kind are not allowedReturns: anumpy.ndarraycontaining the convolved imagesubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 4-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_channels = __import__('4-convolve_channels').convolve_channels


if __name__ == '__main__':

    dataset = np.load('animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
    images_conv = convolve_channels(images, kernel, padding='valid')
    print(images_conv.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0])
    plt.show()
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ ./4-main.py 
(10000, 32, 32, 3)
(10000, 30, 30)Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/convolutions_and_poolingFile:4-convolve_channels.pyHelp×Students who are done with "4. Convolution with Channels"Review your work×Correction of "4. Convolution with Channels"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

5. Multiple KernelsmandatoryWrite a functiondef convolve(images, kernels, padding='same', stride=(1, 1)):that performs a convolution on images using multiple kernels:imagesis anumpy.ndarraywith shape(m, h, w, c)containing multiple imagesmis the number of imageshis the height in pixels of the imageswis the width in pixels of the imagescis the number of channels in the imagekernelsis anumpy.ndarraywith shape(kh, kw, c, nc)containing the kernels for the convolutionkhis the height of a kernelkwis the width of a kernelncis the number of kernelspaddingis either a tuple of(ph, pw), ‘same’, or ‘valid’if ‘same’, performs a same convolutionif ‘valid’, performs a valid convolutionif a tuple:phis the padding for the height of the imagepwis the padding for the width of the imagethe image should be padded with 0’sstrideis a tuple of(sh, sw)shis the stride for the height of the imageswis the stride for the width of the imageYou are only allowed to use threeforloops; any other loops of any kind are not allowedReturns: anumpy.ndarraycontaining the convolved imagesubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 5-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve = __import__('5-convolve').convolve


if __name__ == '__main__':

    dataset = np.load('animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
                       [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0], [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
                       [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])

    images_conv = convolve(images, kernels, padding='valid')
    print(images_conv.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 1])
    plt.show()
    plt.imshow(images_conv[0, :, :, 2])
    plt.show()
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ ./5-main.py 
(10000, 32, 32, 3)
(10000, 30, 30, 3)Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/convolutions_and_poolingFile:5-convolve.pyHelp×Students who are done with "5. Multiple Kernels"Review your work×Correction of "5. Multiple Kernels"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

6. PoolingmandatoryWrite a functiondef pool(images, kernel_shape, stride, mode='max'):that performs pooling on images:imagesis anumpy.ndarraywith shape(m, h, w, c)containing multiple imagesmis the number of imageshis the height in pixels of the imageswis the width in pixels of the imagescis the number of channels in the imagekernel_shapeis a tuple of(kh, kw)containing the kernel shape for the poolingkhis the height of the kernelkwis the width of the kernelstrideis a tuple of(sh, sw)shis the stride for the height of the imageswis the stride for the width of the imagemodeindicates the type of poolingmaxindicates max poolingavgindicates average poolingYou are only allowed to use twoforloops; any other loops of any kind are not allowedReturns: anumpy.ndarraycontaining the pooled imagesubuntu@alexa-ml:~/math/convolutions_and_pooling$ cat 6-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pool = __import__('6-pool').pool


if __name__ == '__main__':

    dataset = np.load('animals_1.npz')
    images = dataset['data']
    print(images.shape)
    images_pool = pool(images, (2, 2), (2, 2), mode='avg')
    print(images_pool.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_pool[0] / 255)
    plt.show()
ubuntu@alexa-ml:~/math/convolutions_and_pooling$ ./6-main.py 
(10000, 32, 32, 3)
(10000, 16, 16, 3)Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/convolutions_and_poolingFile:6-pool.pyHelp×Students who are done with "6. Pooling"Review your work×Correction of "6. Pooling"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Convolutions_and_Pooling.md`
