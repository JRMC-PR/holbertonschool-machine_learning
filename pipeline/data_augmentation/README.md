
## Data Augmentation

### Description
0. FlipmandatoryWrite a functiondef flip_image(image):that flips an image horizontally:imageis a 3Dtf.Tensorcontaining the image to flipReturns the flipped image$ cat 0-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
flip_image = __import__('0-flip').flip_image

tf.random.set_seed(0)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(flip_image(image))
    plt.show()
$ ./0-main.pyRepo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/data_augmentationFile:0-flip.pyHelp×Students who are done with "0. Flip"Review your work×Correction of "0. Flip"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

1. CropmandatoryWrite a functiondef crop_image(image, size):that performs a random crop of an image:imageis a 3Dtf.Tensorcontaining the image to cropsizeis a tuple containing the size of the cropReturns the cropped image$ cat 1-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
crop_image = __import__('1-crop').crop_image

tf.random.set_seed(1)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(crop_image(image, (200, 200, 3)))
    plt.show()
$ ./1-main.pyRepo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/data_augmentationFile:1-crop.pyHelp×Students who are done with "1. Crop"Review your work×Correction of "1. Crop"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

2. RotatemandatoryWrite a functiondef rotate_image(image):that rotates an image by 90 degrees counter-clockwise:imageis a 3Dtf.Tensorcontaining the image to rotateReturns the rotated image$ cat 2-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
rotate_image = __import__('2-rotate').rotate_image

tf.random.set_seed(2)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(rotate_image(image))
    plt.show()
$ ./2-main.pyRepo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/data_augmentationFile:2-rotate.pyHelp×Students who are done with "2. Rotate"Review your work×Correction of "2. Rotate"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

3. ContrastmandatoryWrite a functiondef change_contrast(image, lower, upper):that randomly adjusts the contrast of an image.image: A 3Dtf.Tensorrepresenting the input image to adjust the contrast.lower: A float representing the lower bound of the random contrast factor range.upper: A float representing the upper bound of the random contrast factor range.Returns the contrast-adjusted image.$ cat 3-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_contrast = __import__('3-contrast').change_contrast

tf.random.set_seed(0)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_contrast(image, 0.5, 3))
    plt.show()
$ ./3-main.pyRepo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/data_augmentationFile:3-contrast.pyHelp×Students who are done with "3. Contrast"Review your work×Correction of "3. Contrast"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

4. BrightnessmandatoryWrite a functiondef change_brightness(image, max_delta):that randomly changes the brightness of an image:imageis a 3Dtf.Tensorcontaining the image to changemax_deltais the maximum amount the image should be brightened (or darkened)Returns the altered image$ cat 4-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_brightness = __import__('4-brightness').change_brightness

tf.random.set_seed(4)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_brightness(image, 0.3))
    plt.show()
$ ./4-main.pyRepo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/data_augmentationFile:4-brightness.pyHelp×Students who are done with "4. Brightness"Review your work×Correction of "4. Brightness"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

5. HuemandatoryWrite a functiondef change_hue(image, delta):that changes the hue of an image:imageis a 3Dtf.Tensorcontaining the image to changedeltais the amount the hue should changeReturns the altered image$ cat 5-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_hue = __import__('5-hue').change_hue

tf.random.set_seed(5)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_hue(image, -0.5))
    plt.show()
$ ./5-main.pyRepo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/data_augmentationFile:5-hue.pyHelp×Students who are done with "5. Hue"Review your work×Correction of "5. Hue"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

6. AutomationmandatoryWrite a blog post describing step by step how to perform automated data augmentation. Try to explain every step you know of, and give examples. A total beginner should understand what you have written.Have at least one picture, at the top of the blog postPublish your blog post on Medium or LinkedInShare your blog post at least on LinkedInWrite professionally and intelligiblyPlease, remember that these blogs must be written in English to further your technical ability in a variety of settingsRemember, future employers will see your articles; take this seriously, and produce something that will be an asset to your futureWhen done, please add all urls below (blog post, LinkedIn post, etc.)Add URLs here:SaveHelp×Students who are done with "6. Automation"0/5pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Data_Augmentation.md`
