# Classification Using Neural Networks

## General
## Machine Learning Glossary

- **Model**: In machine learning, a model is a mathematical representation of a real-world process. It is created through a process of training on a dataset.

- **Supervised Learning**: Supervised learning is a type of machine learning where the model is trained on a labeled dataset. The model learns to predict the label from the features of the data.

- **Prediction**: A prediction is the output of a machine learning model when provided with an input instance. It is the model's best guess of the target variable.

- **Node**: In the context of neural networks, a node (or neuron) is a basic unit of a neural network that receives inputs and produces an output.

- **Weight**: Weights are the coefficients that the neural network learns during training. They determine the importance of the input features for the prediction task.

- **Bias**: Bias is an additional parameter in the neural network which is used to adjust the output along with the weighted sum of the inputs to the neuron.

- **Activation Functions**: Activation functions are mathematical equations that determine the output of a neural network. They introduce non-linear properties to the network.

    - **Sigmoid**: The sigmoid function is an activation function that outputs a value between 0 and 1. It is often used for binary classification problems.

    - **Tanh**: The tanh (hyperbolic tangent) function is an activation function that outputs a value between -1 and 1. It is similar to the sigmoid function but can handle negative input values.

    - **Relu**: The ReLU (Rectified Linear Unit) function is an activation function that outputs the input directly if it is positive, otherwise, it outputs zero.

    - **Softmax**: The softmax function is an activation function that turns numbers aka logits into probabilities that sum to one. It is often used in multi-class classification problems.

- **Layer**: A layer in a neural network is a collection of neurons which process a set of input features and produce an output.

- **Hidden Layer**: Hidden layers in a neural network are layers that are not directly connected to the input or output. They perform complex computations on the inputs received from the previous layers.

- **Logistic Regression**: Logistic Regression is a statistical model used for binary classification problems. It uses the logistic sigmoid function as its activation function.

- **Loss Function**: A loss function measures the disparity between the actual and predicted values in machine learning. It is used to update the weights during training.

- **Cost Function**: A cost function is the average of the loss functions of the entire dataset. It is a measure of how well the neural network is performing.

- **Forward Propagation**: Forward propagation is the process of passing the input data through the neural network to get the predicted output.

- **Gradient Descent**: Gradient Descent is an optimization algorithm used to minimize the cost function by iteratively adjusting the model's parameters.

- **Back Propagation**: Backpropagation is the method used to calculate the gradient of the loss function with respect to the weights and biases in a neural network.

- **Computation Graph**: A computation graph is a way to represent a math function in the context of machine learning. It is used in the backpropagation process to compute the gradients.

- **Initializing Weights/Biases**: Weights and biases can be initialized in several ways, such as zero initialization, random initialization, and Xavier/Glorot initialization.

- **Importance of Vectorization**: Vectorization is the process of converting an algorithm from operating on a single value at a time to operating on a set of values (vector) at one time. It is important for computational efficiency.

- **Splitting Data**: Data in machine learning is typically split into a training set, a validation set, and a test set. This is done to evaluate the model's performance and prevent overfitting.

- **Multiclass Classification**: Multiclass classification is a classification task where the output variable can take on more than two values.

- **One-Hot Vector**: A one-hot vector is a vector in which all of the elements are zero, except for one, which is one. It is often used to represent categorical variables.

- **Encoding/Decoding One-Hot Vectors**: One-hot vectors can be encoded using functions like `pandas.get_dummies()` or `sklearn.preprocessing.OneHotEncoder()`. Decoding can be done by finding the index of the maximum value.

- **Softmax Function**: The softmax function is used in the output layer of a neural network, it turns logits into probabilities that sum to one. It is used in multi-class classification problems.

- **Cross-Entropy Loss**: Cross-entropy loss is a loss function that measures the performance of a classification model whose output is a probability value between 0 and 1.

- **Pickling in Python**: Pickling in Python is the process of serializing and deserializing Python object structures. It converts Python objects into a format that can be saved to disk or transmitted over a network.



### Description
0. NeuronmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification:class constructor:def __init__(self, nx):nxis the number of input features to the neuronIfnxis not an integer, raise aTypeErrorwith the exception:nx must be an integerIfnxis less than 1, raise aValueErrorwith the exception:nx must be a positive integerAll exceptions should be raised in the order listed abovePublic instance attributes:W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.b: The bias for the neuron. Upon instantiation, it should be initialized to 0.A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.alexa@ubuntu-xenial:$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('0-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.W.shape)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)
alexa@ubuntu-xenial:$ ./0-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
(1, 784)
0
0
10
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:0-neuron.pyHelp×Students who are done with "0. Neuron"Review your work×Correction of "0. Neuron"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:README.md file exists and is not emptyFile existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalOutput check:nxis 1Output check:nxis a floatOutput check:nxis 0Pycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed7/7pts

1. Privatize NeuronmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification (Based on0-neuron.py):class constructor:def __init__(self, nx):nxis the number of input features to the neuronIfnxis not an integer, raise aTypeErrorwith the exception:nx must be a integerIfnxis less than 1, raise aValueErrorwith the exception:nx must be positiveAll exceptions should be raised in the order listed abovePrivateinstance attributes:__W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.__b: The bias for the neuron. Upon instantiation, it should be initialized to 0.__A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.Each private attribute should have a corresponding getter function (no setter function).alexa@ubuntu-xenial:$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('1-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)
alexa@ubuntu-xenial:$ ./1-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0
0
Traceback (most recent call last):;
  File "./1-main.py", line 16, in <module>
    neuron.A = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:1-neuron.pyHelp×Students who are done with "1. Privatize Neuron"Review your work×Correction of "1. Privatize Neuron"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: All getters workOutput check: all attributes privatized without settersPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

2. Neuron Forward PropagationmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification (Based on1-neuron.py):Add the public methoddef forward_prop(self, X):Calculates the forward propagation of the neuronXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesUpdates the private attribute__AThe neuron should use a sigmoid activation functionReturns the private attribute__Aalexa@ubuntu-xenial:$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('2-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
neuron._Neuron__b = 1
A = neuron.forward_prop(X)
if (A is neuron.A):
        print(A)
alexa@ubuntu-xenial:$ ./2-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]]
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:2-neuron.pyHelp×Students who are done with "2. Neuron Forward Propagation"Review your work×Correction of "2. Neuron Forward Propagation"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalOutput check: output is the same object as the private attributePycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

3. Neuron CostmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification (Based on2-neuron.py):Add the public methoddef cost(self, Y, A):Calculates the cost of the model using logistic regressionYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataAis anumpy.ndarraywith shape (1,m) containing the activated output of the neuron for each exampleTo avoid division by zero errors, please use1.0000001 - Ainstead of1 - AReturns the costalexa@ubuntu-xenial:$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('3-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
cost = neuron.cost(Y, A)
print(cost)
alexa@ubuntu-xenial:$ ./3-main.py
4.365104944262272
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:3-neuron.pyHelp×Students who are done with "3. Neuron Cost"Review your work×Correction of "3. Neuron Cost"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

4. Evaluate NeuronmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification (Based on3-neuron.py):Add the public methoddef evaluate(self, X, Y):Evaluates the neuron’s predictionsXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataReturns the neuron’s prediction and the cost of the network, respectivelyThe prediction should be anumpy.ndarraywith shape (1,m) containing the predicted labels for each exampleThe label values should be 1 if the output of the network is >= 0.5 and 0 otherwisealexa@ubuntu-xenial:$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('4-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A, cost = neuron.evaluate(X, Y)
print(A)
print(cost)
alexa@ubuntu-xenial:$ ./4-main.py
[[0 0 0 ... 0 0 0]]
4.365104944262272
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:4-neuron.pyHelp×Students who are done with "4. Evaluate Neuron"Review your work×Correction of "4. Evaluate Neuron"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

5. Neuron Gradient DescentmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification (Based on4-neuron.py):Add the public methoddef gradient_descent(self, X, Y, A, alpha=0.05):Calculates one pass of gradient descent on the neuronXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataAis anumpy.ndarraywith shape (1,m) containing the activated output of the neuron for each examplealphais the learning rateUpdates the private attributes__Wand__balexa@ubuntu-xenial:$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('5-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b)
alexa@ubuntu-xenial:$ ./5-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0.2579495783615682
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:5-neuron.pyHelp×Students who are done with "5. Neuron Gradient Descent"Review your work×Correction of "5. Neuron Gradient Descent"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: W updated correctlyOutput check: b updated correctlyPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

6. Train NeuronmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification (Based on5-neuron.py):Add the public methoddef train(self, X, Y, iterations=5000, alpha=0.05):Trains the neuronXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataiterationsis the number of iterations to train overifiterationsis not an integer, raise aTypeErrorwith the exceptioniterations must be an integerifiterationsis not positive, raise aValueErrorwith the exceptioniterations must be a positive integeralphais the learning rateifalphais not a float, raise aTypeErrorwith the exceptionalpha must be a floatifalphais not positive, raise aValueErrorwith the exceptionalpha must be positiveAll exceptions should be raised in the order listed aboveUpdates the private attributes__W,__b, and__AYou are allowed to use one loopReturns the evaluation of the training data afteriterationsof training have occurredalexa@ubuntu-xenial:$ cat 6-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('6-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=10)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", np.round(cost, decimals=10))
print("Train accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Train data:", np.round(A, decimals=10))
print("Train Neuron A:", np.round(neuron.A, decimals=10))

A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", np.round(cost, decimals=10))
print("Dev accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Dev data:", np.round(A, decimals=10))
print("Dev Neuron A:", np.round(neuron.A, decimals=10))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()

alexa@ubuntu-xenial:$ ./6-main.py
Train cost: 1.3805076999
Train accuracy: 64.737465456%
Train data: [[0 0 0 ... 0 0 1]]
Train Neuron A: [[2.70000000e-08 2.18229559e-01 1.63492900e-04 ... 4.66530830e-03
  6.06518000e-05 9.73817942e-01]]
Dev cost: 1.4096194345
Dev accuracy: 64.4917257683%
Dev data: [[1 0 0 ... 0 0 1]]
Dev Neuron A: [[0.85021134 0.         0.3526692  ... 0.10140937 0.         0.99555018]]Not that great… Let’s get more data!Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:6-neuron.pyHelp×Students who are done with "6. Train Neuron"Review your work×Correction of "6. Train Neuron"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly one loop allowedOutput check: NormalOutput check: different iterationsOutput check: different alphaOutput check: different iterations and different alphaOutput check: iterations is a floatOutput check: iterations is negativeOutput check: alpha is an integerOutput check: alpha is negativePycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed13/13pts

7. Upgrade Train NeuronmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuronthat defines a single neuron performing binary classification (Based on6-neuron.py):Update the public methodtraintodef train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):Trains the neuron by updating the private attributes__W,__b, and__AXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataiterationsis the number of iterations to train overifiterationsis not an integer, raise aTypeErrorwith the exceptioniterations must be an integerifiterationsis not positive, raise aValueErrorwith the exceptioniterations must be a positive integeralphais the learning rateifalphais not a float, raise aTypeErrorwith the exceptionalpha must be a floatifalphais not positive, raise aValueErrorwith the exceptionalpha must be positiveverboseis a boolean that defines whether or not to print information about the training. IfTrue, printCost after {iteration} iterations: {cost}everystepiterations:Include data from the 0th and last iterationgraphis a boolean that defines whether or not to graph information about the training once the training has completed. IfTrue:Plot the training data everystepiterations as a blue lineLabel the x-axis asiterationLabel the y-axis ascostTitle the plotTraining CostInclude data from the 0th and last iterationOnly if eitherverboseorgraphareTrue:ifstepis not an integer, raise aTypeErrorwith the exceptionstep must be an integerifstepis not positive or is greater thaniterations, raise aValueErrorwith the exceptionstep must be positive and <= iterationsAll exceptions should be raised in the order listed aboveThe 0th iteration should represent the state of the neuron before any training has occurredYou are allowed to use one loopYou can useimport matplotlib.pyplot as pltReturns: the evaluation of the training data afteriterationsof training have occurredalexa@ubuntu-xenial:$ cat 7-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=3000)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./7-main.py
Cost after 0 iterations: 4.365104944262272
Cost after 100 iterations: 0.11955134491351888

...

Cost after 3000 iterations: 0.013386353289868338Train cost: 0.013386353289868338
Train accuracy: 99.66837741808132%
Dev cost: 0.010803484515167197
Dev accuracy: 99.81087470449172%Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:7-neuron.pyHelp×Students who are done with "7. Upgrade Train Neuron"14/14pts

8. NeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification:class constructor:def __init__(self, nx, nodes):nxis the number of input featuresIfnxis not an integer, raise aTypeErrorwith the exception:nx must be an integerIfnxis less than 1, raise aValueErrorwith the exception:nx must be a positive integernodesis the number of nodes found in the hidden layerIfnodesis not an integer, raise aTypeErrorwith the exception:nodes must be an integerIfnodesis less than 1, raise aValueErrorwith the exception:nodes must be a positive integerAll exceptions should be raised in the order listed abovePublic instance attributes:W1: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.b1: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.A1: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.W2: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.b2: The bias for the output neuron. Upon instantiation, it should be initialized to 0.A2: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.alexa@ubuntu-xenial:$ cat 8-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('8-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
print(nn.W1)
print(nn.W1.shape)
print(nn.b1)
print(nn.W2)
print(nn.W2.shape)
print(nn.b2)
print(nn.A1)
print(nn.A2)
nn.A1 = 10
print(nn.A1)
alexa@ubuntu-xenial:$ ./8-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
(3, 784)
[[0.]
 [0.]
 [0.]]
[[ 1.06160017 -1.18488744 -1.80525169]]
(1, 3)
0
0
0
10
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:8-neural_network.pyHelp×Students who are done with "8. NeuralNetwork"Review your work×Correction of "8. NeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalOutput check:nxis 1Output check:nxis a floatOutput check:nxis 0Output check:nodesis 1Output check:nodesis a floatOutput check:nodesis 0Pycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed12/12pts

9. Privatize NeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification (based on8-neural_network.py):class constructor:def __init__(self, nx, nodes):nxis the number of input featuresIfnxis not an integer, raise aTypeErrorwith the exception:nx must be an integerIfnxis less than 1, raise aValueErrorwith the exception:nx must be a positive integernodesis the number of nodes found in the hidden layerIfnodesis not an integer, raise aTypeErrorwith the exception:nodes must be an integerIfnodesis less than 1, raise aValueErrorwith the exception:nodes must be a positive integerAll exceptions should be raised in the order listed abovePrivateinstance attributes:W1: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.b1: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.A1: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.W2: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.b2: The bias for the output neuron. Upon instantiation, it should be initialized to 0.A2: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.Each private attribute should have a corresponding getter function (no setter function).alexa@ubuntu-xenial:$ cat 9-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('9-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
print(nn.A1)
print(nn.A2)
nn.A1 = 10
print(nn.A1)
alexa@ubuntu-xenial:$ ./9-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
[[0.]
 [0.]
 [0.]]
[[ 1.06160017 -1.18488744 -1.80525169]]
0
0
0
Traceback (most recent call last):
  File "./9-main.py", line 19, in <module>
    nn.A1 = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:9-neural_network.pyHelp×Students who are done with "9. Privatize NeuralNetwork"Review your work×Correction of "9. Privatize NeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: All getters workOutput check: all attributes privatized without settersPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

10. NeuralNetwork Forward PropagationmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification (based on9-neural_network.py):Add the public methoddef forward_prop(self, X):Calculates the forward propagation of the neural networkXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesUpdates the private attributes__A1and__A2The neurons should use a sigmoid activation functionReturns the private attributes__A1and__A2, respectivelyalexa@ubuntu-xenial:$ cat 10-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('10-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
nn._NeuralNetwork__b1 = np.ones((3, 1))
nn._NeuralNetwork__b2 = 1
A1, A2 = nn.forward_prop(X)
if A1 is nn.A1:
        print(A1)
if A2 is nn.A2:
        print(A2)
alexa@ubuntu-xenial:$ ./10-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]
 [9.99652394e-01 9.99999995e-01 6.77919152e-01 ... 1.00000000e+00
  9.99662771e-01 9.99990554e-01]
 [5.57969669e-01 2.51645047e-02 4.04250047e-04 ... 1.57024117e-01
  9.97325173e-01 7.41310459e-02]]
[[0.23294587 0.44286405 0.54884691 ... 0.38502756 0.12079644 0.593269  ]]
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:10-neural_network.pyHelp×Students who are done with "10. NeuralNetwork Forward Propagation"Review your work×Correction of "10. NeuralNetwork Forward Propagation"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalOutput check: outputs are the same object as the private attributesPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

11. NeuralNetwork CostmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification (based on10-neural_network.py):Add the public methoddef cost(self, Y, A):Calculates the cost of the model using logistic regressionYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataAis anumpy.ndarraywith shape (1,m) containing the activated output of the neuron for each exampleTo avoid division by zero errors, please use1.0000001 - Ainstead of1 - AReturns the costalexa@ubuntu-xenial:$ cat 11-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('11-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
_, A = nn.forward_prop(X)
cost = nn.cost(Y, A)
print(cost)
alexa@ubuntu-xenial:$ ./11-main.py
0.7917984405648547
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:11-neural_network.pyHelp×Students who are done with "11. NeuralNetwork Cost"Review your work×Correction of "11. NeuralNetwork Cost"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

12. Evaluate NeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification (based on11-neural_network.py):Add the public methoddef evaluate(self, X, Y):Evaluates the neural network’s predictionsXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataReturns the neuron’s prediction and the cost of the network, respectivelyThe prediction should be anumpy.ndarraywith shape (1,m) containing the predicted labels for each exampleThe label values should be 1 if the output of the network is >= 0.5 and 0 otherwisealexa@ubuntu-xenial:$ cat 12-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('12-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A, cost = nn.evaluate(X, Y)
print(A)
print(cost)
alexa@ubuntu-xenial:$ ./12-main.py
[[0 0 0 ... 0 0 0]]
0.7917984405648547
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:12-neural_network.pyHelp×Students who are done with "12. Evaluate NeuralNetwork"Review your work×Correction of "12. Evaluate NeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: NormalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

13. NeuralNetwork Gradient DescentmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification (based on12-neural_network.py):Add the public methoddef gradient_descent(self, X, Y, A1, A2, alpha=0.05):Calculates one pass of gradient descent on the neural networkXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataA1is the output of the hidden layerA2is the predicted outputalphais the learning rateUpdates the private attributes__W1,__b1,__W2, and__b2alexa@ubuntu-xenial:$ cat 13-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('13-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A1, A2 = nn.forward_prop(X)
nn.gradient_descent(X, Y, A1, A2, 0.5)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
alexa@ubuntu-xenial:$ ./13-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
[[ 0.003193  ]
 [-0.01080922]
 [-0.01045412]]
[[ 1.06583858 -1.06149724 -1.79864091]]
[[0.15552509]]
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:13-neural_network.pyHelp×Students who are done with "13. NeuralNetwork Gradient Descent"Review your work×Correction of "13. NeuralNetwork Gradient Descent"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNo loops allowedOutput check: W1 updated correctlyOutput check: W2 updated correctlyPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

14. Train NeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification (based on13-neural_network.py):Add the public methoddef train(self, X, Y, iterations=5000, alpha=0.05):Trains the neural networkXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataiterationsis the number of iterations to train overifiterationsis not an integer, raise aTypeErrorwith the exceptioniterations must be an integerifiterationsis not positive, raise aValueErrorwith the exceptioniterations must be a positive integeralphais the learning rateifalphais not a float, raise aTypeErrorwith the exceptionalpha must be a floatifalphais not positive, raise aValueErrorwith the exceptionalpha must be positiveAll exceptions should be raised in the order listed aboveUpdates the private attributes__W1,__b1,__A1,__W2,__b2, and__A2You are allowed to use one loopReturns the evaluation of the training data afteriterationsof training have occurredalexa@ubuntu-xenial:$ cat 14-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('14-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train, iterations=100)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./14-main.py
Train cost: 0.4680930945144984
Train accuracy: 84.69009080142123%
Dev cost: 0.45985938789496067
Dev accuracy: 86.52482269503547%
alexa@ubuntu-xenial:$Pretty good… but there are still some incorrect labels. We need more data to see why…Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:14-neural_network.pyHelp×Students who are done with "14. Train NeuralNetwork"Review your work×Correction of "14. Train NeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly one loop allowedOutput check: NormalOutput check: different iterationsOutput check: different alphaOutput check: different iterations and different alphaOutput check: iterations is a floatOutput check: iterations is negativeOutput check: alpha is an integerOutput check: alpha is negativePycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed13/13pts

15. Upgrade Train NeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classNeuralNetworkthat defines a neural network with one hidden layer performing binary classification (based on14-neural_network.py):Update the public methodtraintodef train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):Trains the neural networkXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataiterationsis the number of iterations to train overifiterationsis not an integer, raise aTypeErrorwith the exceptioniterations must be an integerifiterationsis not positive, raise aValueErrorwith the exceptioniterations must be a positive integeralphais the learning rateifalphais not a float, raise aTypeErrorwith the exceptionalpha must be a floatifalphais not positive, raise aValueErrorwith the exceptionalpha must be positiveUpdates the private attributes__W1,__b1,__A1,__W2,__b2, and__A2verboseis a boolean that defines whether or not to print information about the training. IfTrue, printCost after {iteration} iterations: {cost}everystepiterations:Include data from the 0th and last iterationgraphis a boolean that defines whether or not to graph information about the training once the training has completed. IfTrue:Plot the training data everystepiterations as a blue lineLabel the x-axis asiterationLabel the y-axis ascostTitle the plotTraining CostInclude data from the 0th and last iterationOnly if eitherverboseorgraphareTrue:ifstepis not an integer, raise aTypeErrorwith the exceptionstep must be an integerifstepis not positive and less than or equal toiterations, raise aValueErrorwith the exceptionstep must be positive and <= iterationsAll exceptions should be raised in the order listed aboveThe 0th iteration should represent the state of the neuron before any training has occurredYou are allowed to use one loopReturns the evaluation of the training data afteriterationsof training have occurredalexa@ubuntu-xenial:$ cat 15-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('15-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./15-main.py
Cost after 0 iterations: 0.7917984405648547
Cost after 100 iterations: 0.4680930945144984

...

Cost after 5000 iterations: 0.024369225667283875Train cost: 0.024369225667283875
Train accuracy: 99.3999210422424%
Dev cost: 0.020330639788072768
Dev accuracy: 99.57446808510639%Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:15-neural_network.pyHelp×Students who are done with "15. Upgrade Train NeuralNetwork"14/14pts

16. DeepNeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification:class constructor:def __init__(self, nx, layers):nxis the number of input featuresIfnxis not an integer, raise aTypeErrorwith the exception:nx must be an integerIfnxis less than 1, raise aValueErrorwith the exception:nx must be a positive integerlayersis a list representing the number of nodes in each layer of the networkIflayersis not a list or an empty list, raise aTypeErrorwith the exception:layers must be a list of positive integersThe first value inlayersrepresents the number of nodes in the first layer, …If the elements inlayersare not all positive integers, raise aTypeErrorwith the exceptionlayers must be a list of positive integersAll exceptions should be raised in the order listed aboveSets the public instance attributes:L: The number of layers in the neural network.cache: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.weights: A dictionary to hold all weights and biased of the network. Upon instantiation:The weights of the network should be initialized using theHe et al.method and saved in theweightsdictionary using the keyW{l}where{l}is the hidden layer the weight belongs toThe biases of the network should be initialized to 0’s and saved in theweightsdictionary using the keyb{l}where{l}is the hidden layer the bias belongs toYou are allowed to use one loopalexa@ubuntu-xenial:$ cat 16-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print(deep.L)
alexa@ubuntu-xenial:$ ./16-main.py
{}
{'W1': array([[ 0.0890981 ,  0.02021099,  0.04943373, ...,  0.02632982,
         0.03090699, -0.06775582],
       [ 0.02408701,  0.00749784,  0.02672082, ...,  0.00484894,
        -0.00227857,  0.00399625],
       [ 0.04295829, -0.04238217, -0.05110231, ..., -0.00364861,
         0.01571416, -0.05446546],
       [ 0.05361891, -0.05984585, -0.09117898, ..., -0.03094292,
        -0.01925805, -0.06308145],
       [-0.01667953, -0.04216413,  0.06239623, ..., -0.02024521,
        -0.05159656, -0.02373981]]), 'b1': array([[0.],
       [0.],
       [0.],
       [0.],
       [0.]]), 'W2': array([[ 0.4609219 ,  0.56004008, -1.2250799 , -0.09454199,  0.57799141],
       [-0.16310703,  0.06882082, -0.94578088, -0.30359994,  1.15661914],
       [-0.49841799, -0.9111359 ,  0.09453424,  0.49877298,  0.75503205]]), 'b2': array([[0.],
       [0.],
       [0.]]), 'W3': array([[-0.42271877,  0.18165055,  0.4444639 ]]), 'b3': array([[0.]])}
3
10
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:16-deep_neural_network.pyHelp×Students who are done with "16. DeepNeuralNetwork"Review your work×Correction of "16. DeepNeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly one loop allowedOutput check: NormalOutput check:nxis 1Output check:nxis a floatOutput check:nxis 0Output check:layersis a list of 1 elementOutput check:layersis not a listOutput check:layersis an empty listOutput check:layerscontains an element that is not a positive integerPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed13/13pts

17. Privatize DeepNeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification (based on16-deep_neural_network.py):class constructor:def __init__(self, nx, layers):nxis the number of input featuresIfnxis not an integer, raise aTypeErrorwith the exception:nx must be an integerIfnxis less than 1, raise aValueErrorwith the exception:nx must be a positive integerlayersis a list representing the number of nodes in each layer of the networkIflayersis not a list, raise aTypeErrorwith the exception:layers must be a list of positive integersThe first value inlayersrepresents the number of nodes in the first layer, …If the elements inlayersare not all positive integers, raise a TypeError with the exceptionlayers must be a list of positive integersAll exceptions should be raised in the order listed aboveSets theprivateinstance attributes:__L: The number of layers in the neural network.__cache: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.__weights: A dictionary to hold all weights and biased of the network. Upon instantiation:The weights of the network should be initialized using theHe et al.method and saved in the__weightsdictionary using the keyW{l}where{l}is the hidden layer the weight belongs toThe biases of the network should be initialized to0‘s and saved in the__weightsdictionary using the keyb{l}where{l}is the hidden layer the bias belongs toEach private attribute should have a corresponding getter function (no setter function).You are allowed to use one loopalexa@ubuntu-xenial:$ cat 17-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('17-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print(deep.L)
alexa@ubuntu-xenial:$ ./17-main.py
{}
{'W1': array([[ 0.0890981 ,  0.02021099,  0.04943373, ...,  0.02632982,
         0.03090699, -0.06775582],
       [ 0.02408701,  0.00749784,  0.02672082, ...,  0.00484894,
        -0.00227857,  0.00399625],
       [ 0.04295829, -0.04238217, -0.05110231, ..., -0.00364861,
         0.01571416, -0.05446546],
       [ 0.05361891, -0.05984585, -0.09117898, ..., -0.03094292,
        -0.01925805, -0.06308145],
       [-0.01667953, -0.04216413,  0.06239623, ..., -0.02024521,
        -0.05159656, -0.02373981]]), 'b1': array([[0.],
       [0.],
       [0.],
       [0.],
       [0.]]), 'W2': array([[ 0.4609219 ,  0.56004008, -1.2250799 , -0.09454199,  0.57799141],
       [-0.16310703,  0.06882082, -0.94578088, -0.30359994,  1.15661914],
       [-0.49841799, -0.9111359 ,  0.09453424,  0.49877298,  0.75503205]]), 'b2': array([[0.],
       [0.],
       [0.]]), 'W3': array([[-0.42271877,  0.18165055,  0.4444639 ]]), 'b3': array([[0.]])}
3
Traceback (most recent call last):
  File "./17-main.py", line 16, in <module>
    deep.L = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:17-deep_neural_network.pyHelp×Students who are done with "17. Privatize DeepNeuralNetwork"Review your work×Correction of "17. Privatize DeepNeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly one loop allowedOutput check: All getters workOutput check: all attributes privatized without settersPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

18. DeepNeuralNetwork Forward PropagationmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification (based on17-deep_neural_network.py):Add the public methoddef forward_prop(self, X):Calculates the forward propagation of the neural networkXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesUpdates the private attribute__cache:The activated outputs of each layer should be saved in the__cachedictionary using the keyA{l}where{l}is the hidden layer the activated output belongs toXshould be saved to thecachedictionary using the keyA0All neurons should use a sigmoid activation functionYou are allowed to use one loopReturns the output of the neural network and the cache, respectivelyalexa@ubuntu-xenial:$ cat 18-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('18-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
deep._DeepNeuralNetwork__weights['b1'] = np.ones((5, 1))
deep._DeepNeuralNetwork__weights['b2'] = np.ones((3, 1))
deep._DeepNeuralNetwork__weights['b3'] = np.ones((1, 1))
A, cache = deep.forward_prop(X)
print(A)
print(cache)
print(cache is deep.cache)
print(A is cache['A3'])
alexa@ubuntu-xenial:$ ./18-main.py
[[0.75603476 0.7516025  0.75526716 ... 0.75228888 0.75522853 0.75217069]]
{'A0': array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), 'A1': array([[0.4678435 , 0.64207147, 0.55271425, ..., 0.61718097, 0.56412986,
        0.72751504],
       [0.79441392, 0.87140579, 0.72851107, ..., 0.8898201 , 0.79466389,
        0.82257068],
       [0.72337339, 0.68239373, 0.63526533, ..., 0.7036234 , 0.7770501 ,
        0.69465346],
       [0.65305735, 0.69829955, 0.58646313, ..., 0.73949722, 0.52054315,
        0.73151973],
       [0.67408798, 0.69624537, 0.73084352, ..., 0.70663173, 0.76204175,
        0.72705428]]), 'A2': array([[0.75067742, 0.78319533, 0.77755571, ..., 0.77891002, 0.75847839,
        0.78517215],
       [0.70591081, 0.71159364, 0.7362214 , ..., 0.70845465, 0.72133875,
        0.71090691],
       [0.72032379, 0.69519095, 0.72414599, ..., 0.70067751, 0.71161433,
        0.70420437]]), 'A3': array([[0.75603476, 0.7516025 , 0.75526716, ..., 0.75228888, 0.75522853,
        0.75217069]])}
True
True
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:18-deep_neural_network.pyHelp×Students who are done with "18. DeepNeuralNetwork Forward Propagation"Review your work×Correction of "18. DeepNeuralNetwork Forward Propagation"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly two loops allowed in totalOutput check: NormalOutput check: outputs are the same object as the attributesPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

19. DeepNeuralNetwork CostmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification (based on18-deep_neural_network.py):Add the public methoddef cost(self, Y, A):Calculates the cost of the model using logistic regressionYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataAis anumpy.ndarraywith shape (1,m) containing the activated output of the neuron for each exampleTo avoid division by zero errors, please use1.0000001 - Ainstead of1 - AReturns the costalexa@ubuntu-xenial:$ cat 19-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('19-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, _ = deep.forward_prop(X)
cost = deep.cost(Y, A)
print(cost)
alexa@ubuntu-xenial:$ ./19-main.py
0.6958649419170609
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:19-deep_neural_network.pyHelp×Students who are done with "19. DeepNeuralNetwork Cost"Review your work×Correction of "19. DeepNeuralNetwork Cost"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly two loops allowed in totalOutput check: NormalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

20. Evaluate DeepNeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification (based on19-deep_neural_network.py):Add the public methoddef evaluate(self, X, Y):Evaluates the neural network’s predictionsXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataReturns the neuron’s prediction and the cost of the network, respectivelyThe prediction should be anumpy.ndarraywith shape (1,m) containing the predicted labels for each exampleThe label values should be 1 if the output of the network is >= 0.5 and 0 otherwisealexa@ubuntu-xenial:$ cat 20-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('20-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, cost = deep.evaluate(X, Y)
print(A)
print(cost)
alexa@ubuntu-xenial:$ ./20-main.py
[[1 1 1 ... 1 1 1]]
0.6958649419170609
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:20-deep_neural_network.pyHelp×Students who are done with "20. Evaluate DeepNeuralNetwork"Review your work×Correction of "20. Evaluate DeepNeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly two loops allowed in totalOutput check: NormalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

21. DeepNeuralNetwork Gradient DescentmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification (based on20-deep_neural_network.py):Add the public methoddef gradient_descent(self, Y, cache, alpha=0.05):Calculates one pass of gradient descent on the neural networkYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input datacacheis a dictionary containing all the intermediary values of the networkalphais the learning rateUpdates the private attribute__weightsYou are allowed to use one loopalexa@ubuntu-xenial:$ cat 21-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('21-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, cache = deep.forward_prop(X)
deep.gradient_descent(Y, cache, 0.5)
print(deep.weights)
alexa@ubuntu-xenial:$ ./21-main.py
{'W1': array([[ 0.0890981 ,  0.02021099,  0.04943373, ...,  0.02632982,
         0.03090699, -0.06775582],
       [ 0.02408701,  0.00749784,  0.02672082, ...,  0.00484894,
        -0.00227857,  0.00399625],
       [ 0.04295829, -0.04238217, -0.05110231, ..., -0.00364861,
         0.01571416, -0.05446546],
       [ 0.05361891, -0.05984585, -0.09117898, ..., -0.03094292,
        -0.01925805, -0.06308145],
       [-0.01667953, -0.04216413,  0.06239623, ..., -0.02024521,
        -0.05159656, -0.02373981]]), 'b1': array([[-1.01835520e-03],
       [-1.22929756e-04],
       [ 9.25521878e-05],
       [ 1.07730873e-04],
       [ 2.29014796e-04]]), 'W2': array([[ 0.4586347 ,  0.55968571, -1.22435332, -0.09516874,  0.57668454],
       [-0.16209305,  0.06902405, -0.9460547 , -0.30329296,  1.15722071],
       [-0.49595566, -0.91068385,  0.09382566,  0.49948968,  0.75647764]]), 'b2': array([[-0.00055419],
       [ 0.00032369],
       [ 0.0007201 ]]), 'W3': array([[-0.41262664,  0.18889024,  0.44717929]]), 'b3': array([[0.00659936]])}
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:21-deep_neural_network.pyHelp×Students who are done with "21. DeepNeuralNetwork Gradient Descent"Review your work×Correction of "21. DeepNeuralNetwork Gradient Descent"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly three loops allowed in totalOutput check: NormalOutput check: different alphaPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed6/6pts

22. Train DeepNeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification (based on21-deep_neural_network.py):Add the public methoddef train(self, X, Y, iterations=5000, alpha=0.05):Trains the deep neural networkXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataiterationsis the number of iterations to train overifiterationsis not an integer, raise aTypeErrorwith the exceptioniterations must be an integerifiterationsis not positive, raise aValueErrorwith the exceptioniterations must be a positive integeralphais the learning rateifalphais not a float, raise a TypeError with the exceptionalpha must be a floatifalphais not positive, raise a ValueError with the exceptionalpha must be positiveAll exceptions should be raised in the order listed aboveUpdates the private attributes__weightsand__cacheYou are allowed to use one loopReturns the evaluation of the training data afteriterationsof training have occurredalexa@ubuntu-xenial:$ cat 22-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('22-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [5, 3, 1])
A, cost = deep.train(X_train, Y_train, iterations=100)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./22-main.py
Train cost: 0.6444304786060048
Train accuracy: 56.241610738255034%
Dev cost: 0.6428913158565179
Dev accuracy: 57.730496453900706%Hmm… doesn’t seem like this worked very well. Could it be because of our architecture or that it wasn’t trained properly? We need to see more information…Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:22-deep_neural_network.pyHelp×Students who are done with "22. Train DeepNeuralNetwork"Review your work×Correction of "22. Train DeepNeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOnly four loops allowed in totalOutput check: NormalOutput check: different iterationsOutput check: different alphaOutput check: different iterations and different alphaOutput check: iterations is a floatOutput check: iterations is negativeOutput check: alpha is an integerOutput check: alpha is negativePycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed13/13pts

23. Upgrade Train DeepNeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Write a classDeepNeuralNetworkthat defines a deep neural network performing binary classification (based on22-deep_neural_network.py):Update the public methodtraintodef train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):Trains the deep neural network by updating the private attributes__weightsand__cacheXis anumpy.ndarraywith shape (nx,m) that contains the input datanxis the number of input features to the neuronmis the number of examplesYis anumpy.ndarraywith shape (1,m) that contains the correct labels for the input dataiterationsis the number of iterations to train overifiterationsis not an integer, raise aTypeErrorwith the exceptioniterations must be an integerifiterationsis not positive, raise aValueErrorwith the exceptioniterations must be a positive integeralphais the learning rateifalphais not a float, raise aTypeErrorwith the exceptionalpha must be a floatifalphais not positive, raise aValueErrorwith the exceptionalpha must be positiveverboseis a boolean that defines whether or not to print information about the training. IfTrue, printCost after {iteration} iterations: {cost}everystepiterations:Include data from the 0th and last iterationgraphis a boolean that defines whether or not to graph information about the training once the training has completed. IfTrue:Plot the training data everystepiterations as a blue lineLabel the x-axis asiterationLabel the y-axis ascostTitle the plotTraining CostInclude data from the 0th and last iterationOnly if eitherverboseorgraphareTrue:ifstepis not an integer, raise aTypeErrorwith the exceptionstep must be an integerifstepis not positive and less than or equal toiterations, raise aValueErrorwith the exceptionstep must be positive and <= iterationsAll exceptions should be raised in the order listed aboveThe 0th iteration should represent the state of the neuron before any training has occurredYou are allowed to use one loopReturns the evaluation of the training data afteriterationsof training have occurredalexa@ubuntu-xenial:$ cat 23-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('23-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [5, 3, 1])
A, cost = deep.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./23-main.py
Cost after 0 iterations: 0.6958649419170609
Cost after 100 iterations: 0.6444304786060048

...
Cost after 4800 iterations: 0.012130338888226167
Cost after 4900 iterations: 0.011896856912322803
Cost after 5000 iterations: 0.011671820326008163Train cost: 0.011671820326008163
Train accuracy: 99.88945913936044%
Dev cost: 0.009249552132279246
Dev accuracy: 99.95271867612293%Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:23-deep_neural_network.pyHelp×Students who are done with "23. Upgrade Train DeepNeuralNetwork"14/14pts

24. One-Hot EncodemandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef one_hot_encode(Y, classes):that converts a numeric label vector into a one-hot matrix:Yis anumpy.ndarraywith shape (m,) containing numeric class labelsmis the number of examplesclassesis the maximum number of classes found inYReturns: a one-hot encoding ofYwith shape (classes,m), orNoneon failurealexa@ubuntu-xenial:$ cat 24-main.py
#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)
alexa@ubuntu-xenial:$ ./24-main.py
[5 0 4 1 9 2 1 3 1 4]
[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 1. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:24-one_hot_encode.pyHelp×Students who are done with "24. One-Hot Encode"Review your work×Correction of "24. One-Hot Encode"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Output check: NormalOutput check: smallest element inYis greater than 0Output check:classesis larger than largest element inYOutput check:Yis not anumpy.ndarrayOutput check:classesis not an integerOutput check:classesis less than 2Output check:classesis smaller than largest element inYPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed22/22pts

25. One-Hot DecodemandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef one_hot_decode(one_hot):that converts a one-hot matrix into a vector of labels:one_hotis a one-hot encodednumpy.ndarraywith shape (classes,m)classesis the maximum number of classesmis the number of examplesReturns: anumpy.ndarraywith shape (m, ) containing the numeric labels for each example, orNoneon failurealexa@ubuntu-xenial:$ cat 25-main.py
#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode
oh_decode = __import__('25-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)
alexa@ubuntu-xenial:$ ./25-main.py
[5 0 4 1 9 2 1 3 1 4]
[5 0 4 1 9 2 1 3 1 4]
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:25-one_hot_decode.pyHelp×Students who are done with "25. One-Hot Decode"Review your work×Correction of "25. One-Hot Decode"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Output check: NormalOutput check:ohis not anumpy.ndarrayOutput check:oh.ndimis not 2Pycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed16/16pts

26. Persistence is KeymandatoryScore:100.00%(Checks completed: 100.00%)Update the classDeepNeuralNetwork(based on23-deep_neural_network.py):Create the instance methoddef save(self, filename):Saves the instance object to a file inpickleformatfilenameis the file to which the object should be savedIffilenamedoes not have the extension.pkl, add itCreate the static methoddef load(filename):Loads a pickledDeepNeuralNetworkobjectfilenameis the file from which the object should be loadedReturns: the loaded object, orNoneiffilenamedoesn’t existalexa@ubuntu-xenial:$ cat 26-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('26-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [3, 1])
A, cost = deep.train(X_train, Y_train, iterations=400, graph=False)
deep.save('26-output')
del deep

saved = Deep.load('26-output.pkl')
A_saved, cost_saved = saved.evaluate(X_train, Y_train)

print(np.array_equal(A, A_saved) and cost == cost_saved)
alexa@ubuntu-xenial:$ ls 26-output*
ls: cannot access '26-output*': No such file or directory
alexa@ubuntu-xenial:$ ./26-main.py
Cost after 0 iterations: 0.7773240521521816
Cost after 100 iterations: 0.18751378071323063
Cost after 200 iterations: 0.12117095705345622
Cost after 300 iterations: 0.09031067302785326
Cost after 400 iterations: 0.07222364349190777
True
alexa@ubuntu-xenial:$ ls 26-output*
26-output.pkl
alexa@ubuntu-xenial:$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:26-deep_neural_network.pyHelp×Students who are done with "26. Persistence is Key"Review your work×Correction of "26. Persistence is Key"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Output check: NormalsaveOutput check:savefile withoutpklextensionOutput check: NormalloadOutput check:loadnonexistantfilenamePycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed26/26pts

27. Update DeepNeuralNetworkmandatoryScore:100.00%(Checks completed: 100.00%)Update the classDeepNeuralNetworkto perform multiclass classification (based on26-deep_neural_network.py):You will need to update the instance methodsforward_prop,cost, andevaluateYis now a one-hotnumpy.ndarrayof shape(classes, m)Ideally, you should not have to change the__init__,gradient_descent, ortraininstance methodsBecause the training process takes such a long time, I have pretrained a model for you to load and finish training (27-saved.pkl). This model has already been trained for 2000 iterations.The training process may take up to 5 minutesalexa@ubuntu-xenial:$ cat 27-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('27-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)

deep = Deep.load('27-saved.pkl')
A_one_hot, cost = deep.train(X_train, Y_train_one_hot, iterations=100,
                             step=10, graph=False)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_train == A) / Y_train.shape[0] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

A_one_hot, cost = deep.evaluate(X_valid, Y_valid_one_hot)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_valid == A) / Y_valid.shape[0] * 100
print("Validation cost:", cost)
print("Validation accuracy: {}%".format(accuracy))

deep.save('27-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_valid_3D[i])
    plt.title(A[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
ubuntu@alexa-ml:~$ ./27-main.py
Cost after 0 iterations: 0.4388904112857043
Cost after 10 iterations: 0.43778288041633584
Cost after 20 iterations: 0.43668839872612714
Cost after 30 iterations: 0.4356067473605946
Cost after 40 iterations: 0.4345377117680657
Cost after 50 iterations: 0.43348108159932525
Cost after 60 iterations: 0.43243665061046194
Cost after 70 iterations: 0.43140421656876826
Cost after 80 iterations: 0.43038358116155134
Cost after 90 iterations: 0.4293745499077263
Cost after 99 iterations: 0.4293745499077263
Train cost: 0.42837693207206456
Train accuracy: 88.442%
Validation cost: 0.39517557351173044
Validation accuracy: 89.64%As you can see, our training has become very slow and is beginning to plateau. Let’s alter the model a little and see if we get a better resultRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:27-deep_neural_network.pyHelp×Students who are done with "27. Update DeepNeuralNetwork"Review your work×Correction of "27. Update DeepNeuralNetwork"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Output check: NormalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed16/16pts

28. All the ActivationsmandatoryScore:100.00%(Checks completed: 100.00%)Update the classDeepNeuralNetworkto allow different activation functions (based on27-deep_neural_network.py):Update the__init__method todef __init__(self, nx, layers, activation='sig'):activationrepresents the type of activation function used in the hidden layerssigrepresents a sigmoid activationtanhrepresents a tanh activationifactivationis notsigortanh, raise aValueErrorwith the exception:activation must be 'sig' or 'tanh'Create the private attribute__activationand set it to the value ofactivationCreate a getter for the private attribute__activationUpdate theforward_propandgradient_descentinstance methods to use the__activationfunction in the hidden layersBecause the training process takes such a long time, I have pre-trained a model for you to load and finish training (28-saved.pkl). This model has already been trained for 2000 iterations.The training process may take up to 5 minutesalexa@ubuntu-xenial:$ cat 28-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep27 = __import__('27-deep_neural_network').DeepNeuralNetwork
Deep28 = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_test_3D = lib['X_test']
Y_test = lib['Y_test']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)
Y_test_one_hot = one_hot_encode(Y_test, 10)

print('Sigmoid activation:')
deep27 = Deep27.load('27-output.pkl')
A_one_hot27, cost27 = deep27.evaluate(X_train, Y_train_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_train == A27) / Y_train.shape[0] * 100
print("Train cost:", cost27)
print("Train accuracy: {}%".format(accuracy27))
A_one_hot27, cost27 = deep27.evaluate(X_valid, Y_valid_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_valid == A27) / Y_valid.shape[0] * 100
print("Validation cost:", cost27)
print("Validation accuracy: {}%".format(accuracy27))
A_one_hot27, cost27 = deep27.evaluate(X_test, Y_test_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_test == A27) / Y_test.shape[0] * 100
print("Test cost:", cost27)
print("Test accuracy: {}%".format(accuracy27))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A27[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

print('\nTanh activaiton:')

deep28 = Deep28.load('28-saved.pkl')
A_one_hot28, cost28 = deep28.train(X_train, Y_train_one_hot, iterations=100,
                                step=10, graph=False)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_train == A28) / Y_train.shape[0] * 100
print("Train cost:", cost28)
print("Train accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_valid, Y_valid_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_valid == A28) / Y_valid.shape[0] * 100
print("Validation cost:", cost28)
print("Validation accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_test, Y_test_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_test == A28) / Y_test.shape[0] * 100
print("Test cost:", cost28)
print("Test accuracy: {}%".format(accuracy28))
deep28.save('28-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A28[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

alexa@ubuntu-xenial:$ ./28-main.py
Sigmoid activation:
Train cost: 0.42837693207206456
Train accuracy: 88.442%
Validation cost: 0.39517557351173044
Validation accuracy: 89.64%
Test cost: 0.4074169894615401
Test accuracy: 89.0%Tanh activaiton:
Cost after 0 iterations: 0.1806181562229199
Cost after 10 iterations: 0.18012009542718574
Cost after 20 iterations: 0.17962428978349268
Cost after 30 iterations: 0.17913072860418566
Cost after 40 iterations: 0.1786394012066576
Cost after 50 iterations: 0.17815029691267448
Cost after 60 iterations: 0.17766340504784378
Cost after 70 iterations: 0.1771787149412177
Cost after 80 iterations: 0.1766962159250237
Cost after 90 iterations: 0.17621589733451382
Train cost: 0.17573774850792664
Train accuracy: 95.006%
Validation cost: 0.1768930960039794
Validation accuracy: 95.13000000000001%
Test cost: 0.1809489808838737
Test accuracy: 94.77%The training of this model is also getting slow and plateauing after about 2000 iterations. However, just by changing the activation function, we have nearly halved the model’s cost and increased its accuracy by about 6%Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/classificationFile:28-deep_neural_network.pyHelp×Students who are done with "28. All the Activations"Review your work×Correction of "28. All the Activations"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Output check: DefaultsigmoidactivationOutput check:tanhactivationOutput check: invalid activationPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed20/20pts

29. Blogpost#advancedScore:0.00%(Checks completed: 0.00%)Write a blog post that explains the purpose of activation functions and compares and contrasts (at the minimum) the following functions:BinaryLinearSigmoidTanhReLUSoftmaxYour posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.When done, please add all URLs below (blog post, LinkedIn post, etc.)Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.Add URLs here:SaveHelp×Students who are done with "29. Blogpost"0/5pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Classification_Using_Neural_Networks.md`
