
## Transfer Learning

### Description
0. Transfer KnowledgemandatoryScore:100.00%(Checks completed: 100.00%)Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:Keras pakages a number of deep leanring models alongside pre-trained weights into an applications module.You must use one of the applications listed inKeras ApplicationsYour script must save your trained model in the current working directory ascifar10.h5Your saved model should be compiledYour saved model should have a validation accuracy of 87% or higherYour script should not run when the file is importedHint1:The training and tweaking of hyperparameters may take a while so start early!Hint2:The CIFAR 10 dataset contains 32x32 pixel images, however most of the Keras applications are trained on much larger images. Your first layer should be a lambda layer that scales up the data to the correct sizeHint3:You will want to freeze most of the application layers. Since these layers will always produce the same output, you should compute the output of the frozen layers ONCE and use those values as input to train the remaining trainable layers. This will save you A LOT of time.In the same file, write a functiondef preprocess_data(X, Y):that pre-processes the data for your model:Xis anumpy.ndarrayof shape(m, 32, 32, 3)containing the CIFAR 10 data, where m is the number of data pointsYis anumpy.ndarrayof shape(m,)containing the CIFAR 10 labels forXReturns:X_p, Y_pX_pis anumpy.ndarraycontaining the preprocessedXY_pis anumpy.ndarraycontaining the preprocessedYNOTE:About half of the points for this project are for the blog post in the next task. While you are attempting to train your model, keep track of what you try and why so that you have a log to reference when it is time to write your report.alexa@ubuntu-xenial:transfer_learning$ cat 0-main.py
#!/usr/bin/env python3

from tensorflow import keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
alexa@ubuntu-xenial:transfer_learning$ ./0-main.py
10000/10000 [==============================] - 159s 16ms/sample - loss: 0.3329 - acc: 0.8864Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/transfer_learningFile:0-transfer.pyHelp×Students who are done with "0. Transfer Knowledge"QA Review×0. Transfer KnowledgeCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---10/10pts

1. "Research is what I'm doing when I don't know what I'm doing." - Wernher von BraunmandatoryScore:100.00%(Checks completed: 100.00%)Write a blog post explaining your experimental process in completing the task above written as a journal-style scientific paper:Experimental processSection of PaperWhat did I do in a nutshell?AbstractWhat is the problem?IntroductionHow did I solve the problem?Materials and MethodsWhat did I find out?ResultsWhat does it mean?DiscussionWho helped me out?Acknowledgments (optional)Whose work did I refer to?Literature CitedExtra InformationAppendices (optional)Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.When done, please add all URLs below (blog post, tweet, etc.)Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.Add URLs here:Savehttps://www.linkedin.com/pulse/cifar-10-scrimmage-when-resnet50-efficientnetb7-step-ring-m%25C3%25A9ndez--dak3f/RemoveHelp×Students who are done with "1. "Research is what I'm doing when I don't know what I'm doing." - Wernher von Braun"QA Review×1. "Research is what I'm doing when I don't know what I'm doing." - Wernher von BraunCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---12/12pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Transfer_Learning.md`
