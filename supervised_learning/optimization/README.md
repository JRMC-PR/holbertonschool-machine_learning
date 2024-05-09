
## Optimization

### Description
0. Normalization ConstantsmandatoryWrite the functiondef normalization_constants(X):that calculates the normalization (standardization) constants of a matrix:Xis thenumpy.ndarrayof shape(m, nx)to normalizemis the number of data pointsnxis the number of featuresReturns: the mean and standard deviation of each feature, respectivelyubuntu@ml:~/optimization$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(m)
    print(s)
ubuntu@ml:~/optimization$ ./0-main.py 
[ 0.11961603  2.08201297 -3.59232261]
[2.01576449 1.034667   9.52002619]
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:0-norm_constants.pyHelp×Students who are done with "0. Normalization Constants"Review your work×Correction of "0. Normalization Constants"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

1. NormalizemandatoryWrite the functiondef normalize(X, m, s):that normalizes (standardizes) a matrix:Xis thenumpy.ndarrayof shape(d, nx)to normalizedis the number of data pointsnxis the number of featuresmis anumpy.ndarrayof shape(nx,)that contains the mean of all features ofXsis anumpy.ndarrayof shape(nx,)that contains the standard deviation of all features ofXReturns: The normalizedXmatrixubuntu@ml:~/optimization$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants
normalize = __import__('1-normalize').normalize

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(X[:10])
    X = normalize(X, m, s)
    print(X[:10])
    m, s = normalization_constants(X)
    print(m)
    print(s)
ubuntu@ml:~/optimization$ ./1-main.py 
[[  3.52810469   3.8831507   -6.69181838]
 [  0.80031442   0.65224094  -5.39379178]
 [  1.95747597   0.729515     7.99659596]
 [  4.4817864    2.96939671   3.55263731]
 [  3.73511598   0.82687659   3.40131526]
 [ -1.95455576   3.94362119 -19.16956044]
 [  1.90017684   1.58638102  -3.24326124]
 [ -0.30271442   1.25254519 -10.38030909]
 [ -0.2064377    3.92294203  -0.20075401]
 [  0.821197     3.48051479  -3.9815039 ]]
[[ 1.69091612  1.74078977 -0.32557639]
 [ 0.33768746 -1.38186686 -0.18922943]
 [ 0.91174338 -1.3071819   1.21732003]
 [ 2.16402779  0.85765153  0.75051893]
 [ 1.79361228 -1.21308245  0.73462381]
 [-1.02897526  1.79923417 -1.63625998]
 [ 0.88331787 -0.47902557  0.03666601]
 [-0.20951378 -0.80167608 -0.71302183]
 [-0.1617519   1.77924787  0.35625623]
 [ 0.34804709  1.35164437 -0.04088028]]
[ 2.44249065e-17 -4.99600361e-16  1.46549439e-16]
[1. 1. 1.]
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:1-normalize.pyHelp×Students who are done with "1. Normalize"Review your work×Correction of "1. Normalize"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

2. Shuffle DatamandatoryWrite the functiondef shuffle_data(X, Y):that shuffles the data points in two matrices the same way:Xis the firstnumpy.ndarrayof shape(m, nx)to shufflemis the number of data pointsnxis the number of features inXYis the secondnumpy.ndarrayof shape(m, ny)to shufflemis the same number of data points as inXnyis the number of features inYReturns: the shuffledXandYmatricesHint: you should usenumpy.random.permutationubuntu@ml:~/optimization$ cat 2-main.py 
#!/usr/bin/env python3

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data

if __name__ == '__main__':
    X = np.array([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8], 
                [9, 10]])
    Y = np.array([[11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
                [19, 20]])

    np.random.seed(0)
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    print(X_shuffled)
    print(Y_shuffled)
ubuntu@ml:~/optimization$ ./2-main.py 
[[ 5  6]
 [ 1  2]
 [ 3  4]
 [ 7  8]
 [ 9 10]]
[[15 16]
 [11 12]
 [13 14]
 [17 18]
 [19 20]]
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:2-shuffle_data.pyHelp×Students who are done with "2. Shuffle Data"Review your work×Correction of "2. Shuffle Data"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

3. Mini-BatchmandatoryWrite a functiondef create_mini_batches(X, Y, batch_size):that creates mini-batches to be used for training a neural network using mini-batch gradient descent:Xis anumpy.ndarrayof shape(m, nx)representing input datamis the number of data pointsnxis the number of features inXYis anumpy.ndarrayof shape(m, ny)representing the labelsmis the same number of data points as inXnyis the number of classes for classification tasks.batch_sizeis the number of data points in a batchReturns: list of mini-batches containing tuples(X_batch, Y_batch)Your function should allow for a smaller final batch (i.e. use the entire dataset)You should useshuffle_data = __import__('2-shuffle_data').shuffle_dataubuntu@ml:~/optimization$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_mini_batches = __import__('3-mini_batch').create_mini_batches


def one_hot(Y, classes):
    """Convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh = one_hot(Y, 10)
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
Y_valid_oh = one_hot(Y_valid, 10)

model = tf.keras.models.load_model('model.h5', compile=False)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

batch_size = 32
epochs = 10

loss_fn = tf.keras.losses.CategoricalCrossentropy()

for epoch in range(epochs):
    print(f"After {epoch} epochs:")

    train_loss = tf.reduce_mean(loss_fn(Y_oh, model(X)))
    train_accuracy = np.mean(np.argmax(model(X), axis=1) == Y)

    valid_loss = tf.reduce_mean(loss_fn(Y_valid_oh, model(X_valid)))
    valid_accuracy = np.mean(np.argmax(model(X_valid), axis=1) == Y_valid)

    print(f"\tTraining Cost: {train_loss}")
    print(f"\tTraining Accuracy: {train_accuracy}")
    print(f"\tValidation Cost: {valid_loss}")
    print(f"\tValidation Accuracy: {valid_accuracy}")

    for step, (X_batch, Y_batch) in enumerate(create_mini_batches(X, Y_oh, batch_size)):
        with tf.GradientTape() as tape:
            predictions = model(X_batch)
            loss = loss_fn(Y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (step + 1) % 100 == 0:
            Y_pred = np.argmax(predictions, axis=1)
            batch_accuracy = np.mean(Y_pred == np.argmax(Y_batch, axis=1))
            print(f"\tStep {step + 1}:")
            print(f"\t\tCost: {loss}")
            print(f"\t\tAccuracy: {batch_accuracy}")

print(f"After {epochs} epochs:")

final_train_loss = tf.reduce_mean(loss_fn(Y_oh, model(X)))
final_train_accuracy = np.mean(np.argmax(model(X), axis=1) == Y)

final_valid_loss = tf.reduce_mean(loss_fn(Y_valid_oh, model(X_valid)))
final_valid_accuracy = np.mean(np.argmax(model(X_valid), axis=1) == Y_valid)

print(f"\tFinal Training Cost: {final_train_loss}, Accuracy: {final_train_accuracy}")
print(f"\tFinal Validation Cost: {final_valid_loss}, Accuracy: {final_valid_accuracy}")

ubuntu@ml:~/optimization$ ./3-main.py 
After 0 epochs:
        Training Cost: 2.3219268321990967
        Training Accuracy: 0.13176
        Validation Cost: 2.318324565887451
        Validation Accuracy: 0.1315
        Step 100:
                Cost: 1.2482386827468872
                Accuracy: 0.625
        Step 200:
                Cost: 0.8422225713729858
                Accuracy: 0.8125

    ...

        Step 1500:
                Cost: 0.37368184328079224
                Accuracy: 0.84375
After 1 epochs:
        Training Cost: 0.3559108078479767
        Training Accuracy: 0.90134
        Validation Cost: 0.3254995048046112
        Validation Accuracy: 0.9095

...

After 2 epochs:
        Training Cost: 0.3029726445674896
        Training Accuracy: 0.9133
        Validation Cost: 0.2842124104499817
        Validation Accuracy: 0.918

...

After 9 epochs:
        Training Cost: 0.17719177901744843
        Training Accuracy: 0.94862
        Validation Cost: 0.17647580802440643
        Validation Accuracy: 0.952
        Step 100:
                Cost: 0.13570508360862732
                Accuracy: 1.0
        Step 200:
                Cost: 0.2039051502943039
                Accuracy: 0.96875
        Step 300:
                Cost: 0.17096975445747375
                Accuracy: 0.90625

    ...

        Step 1500:
                Cost: 0.15383833646774292
                Accuracy: 0.9375
After 10 epochs:
        Final Training Cost: 0.16524291038513184, Accuracy: 0.9523
        Final Validation Cost: 0.16789411008358002, Accuracy: 0.9537
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:3-mini_batch.pyHelp×Students who are done with "3. Mini-Batch"Review your work×Correction of "3. Mini-Batch"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/5pts

4. Moving AveragemandatoryWrite the functiondef moving_average(data, beta):that calculates the weighted moving average of a data set:datais the list of data to calculate the moving average ofbetais the weight used for the moving averageYour moving average calculation should use bias correctionReturns: a list containing the moving averages ofdataubuntu@ml:~/optimization$ cat 4-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
moving_average = __import__('4-moving_average').moving_average

if __name__ == '__main__':
        data = [72, 78, 71, 68, 66, 69, 79, 79, 65, 64, 66, 78, 64, 64, 81, 71, 69,
                65, 72, 64, 60, 61, 62, 66, 72, 72, 67, 67, 67, 68, 75]
        days = list(range(1, len(data) + 1))
        m_avg = moving_average(data, 0.9)
        print(m_avg)
        plt.plot(days, data, 'r', days, m_avg, 'b')
        plt.xlabel('Day of Month')
        plt.ylabel('Temperature (Fahrenheit)')
        plt.title('SF Maximum Temperatures in October 2018')
        plt.legend(['actual', 'moving_average'])
        plt.show()
ubuntu@ml:~/optimization$ ./4-main.py 
[72.0, 75.15789473684211, 73.62361623616238, 71.98836871183484, 70.52604332006544, 70.20035470453027, 71.88706986789997, 73.13597603396988, 71.80782582850702, 70.60905915023126, 69.93737009120935, 71.0609712312634, 70.11422355031073, 69.32143707981284, 70.79208718739721, 70.81760741911772, 70.59946700377961, 69.9406328280786, 70.17873340222755, 69.47534437750306, 68.41139351151023, 67.58929643210207, 66.97601174673004, 66.86995043877324, 67.42263231561797, 67.91198666959514, 67.8151574064495, 67.72913996327617, 67.65262186609462, 67.68889744321645, 68.44900744806469]Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:4-moving_average.pyHelp×Students who are done with "4. Moving Average"Review your work×Correction of "4. Moving Average"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

5. MomentummandatoryWrite the functiondef update_variables_momentum(alpha, beta1, var, grad, v):that updates a variable using the gradient descent with momentum optimization algorithm:alphais the learning ratebeta1is the momentum weightvaris anumpy.ndarraycontaining the variable to be updatedgradis anumpy.ndarraycontaining the gradient ofvarvis the previous first moment ofvarReturns: the updated variable and the new moment, respectivelyubuntu@ml:~/optimization$ cat 5-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_momentum = __import__('5-momentum').update_variables_momentum

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev = np.zeros((nx, 1))
    db_prev = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev = update_variables_momentum(0.01, 0.9, W, dW, dW_prev)
        b, db_prev = update_variables_momentum(0.01, 0.9, b, db, db_prev)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A >= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@ml:~/optimization$ ./5-main.py 
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 0.5729736703124043
Cost after 200 iterations: 0.2449357405113111
Cost after 300 iterations: 0.1771132508758216
Cost after 400 iterations: 0.14286111618067307
Cost after 500 iterations: 0.12051674907075897
Cost after 600 iterations: 0.10450664363662195
Cost after 700 iterations: 0.09245615061035156
Cost after 800 iterations: 0.08308760082979069
Cost after 900 iterations: 0.0756292416282403
Cost after 1000 iterations: 0.0695782354732263Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:5-momentum.pyHelp×Students who are done with "5. Momentum"Review your work×Correction of "5. Momentum"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

6. Momentum UpgradedmandatoryWrite the functiondef create_momentum_op(alpha, beta1):that sets up the gradient descent with momentum optimization algorithm inTensorFlow:alphais the learning rate.beta1is the momentum weight.Returns:optimizerubuntu@ml:~/optimization$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_momentum_op = __import__('6-momentum').create_momentum_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh=one_hot(Y,10)

model = tf.keras.models.load_model('model.h5', compile=False)

optimizer=create_momentum_op(0.01, 0.9)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


total_iterations = 1000
for iteration in range(total_iterations):

    cost = train_step(X, Y_oh)

    if (iteration + 1) % 100 == 0:
        print(f'Cost after {iteration + 1} iterations: {cost}')


Y_pred_oh = model(X[:100])
Y_pred = np.argmax(Y_pred_oh, axis=1)

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(str(Y_pred[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()

ubuntu@ml:~/optimization$ ./6-main.py 
Cost after 100 iterations: 0.4046999216079712
Cost after 200 iterations: 0.33621034026145935
Cost after 300 iterations: 0.3055974543094635
Cost after 400 iterations: 0.2854801118373871
Cost after 500 iterations: 0.2698008716106415
Cost after 600 iterations: 0.2563629746437073
Cost after 700 iterations: 0.24418331682682037
Cost after 800 iterations: 0.23279382288455963
Cost after 900 iterations: 0.2219877392053604
Cost after 1000 iterations: 0.2116965502500534Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:6-momentum.pyHelp×Students who are done with "6. Momentum Upgraded"Review your work×Correction of "6. Momentum Upgraded"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

7. RMSPropmandatoryWrite the functiondef update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):that updates a variable using the RMSProp optimization algorithm:alphais the learning ratebeta2is the RMSProp weightepsilonis a small number to avoid division by zerovaris anumpy.ndarraycontaining the variable to be updatedgradis anumpy.ndarraycontaining the gradient ofvarsis the previous second moment ofvarReturns: the updated variable and the new moment, respectivelyubuntu@ml:~/optimization$ cat 7-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_RMSProp = __import__('7-RMSProp').update_variables_RMSProp

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev = np.zeros((nx, 1))
    db_prev = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev = update_variables_RMSProp(0.001, 0.9, 1e-8, W, dW, dW_prev)
        b, db_prev = update_variables_RMSProp(0.001, 0.9, 1e-8, b, db, db_prev)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A >= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@ml:~/optimization$ ./7-main.py 
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 1.370832184880606
Cost after 200 iterations: 0.2269339299030878
Cost after 300 iterations: 0.0513339480022191
Cost after 400 iterations: 0.018365571163723598
Cost after 500 iterations: 0.008176390663315379
Cost after 600 iterations: 0.004091350591779443
Cost after 700 iterations: 0.00219563959482299
Cost after 800 iterations: 0.0011481585722158702
Cost after 900 iterations: 0.000559930891318418
Cost after 1000 iterations: 0.00026558128741123633Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:7-RMSProp.pyHelp×Students who are done with "7. RMSProp"Review your work×Correction of "7. RMSProp"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

8. RMSProp UpgradedmandatoryWrite the functiondef create_RMSProp_op(loss, alpha, beta2, epsilon):that sets up the RMSProp optimization algorithm inTensorFlow:alphais the learning ratebeta2is the RMSProp weight (Discounting factor)epsilonis a small number to avoid division by zeroReturns:optimizerubuntu@ml:~/optimization$ cat 8-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_RMSProp_op = __import__('8-RMSProp').create_RMSProp_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh=one_hot(Y,10)

model = tf.keras.models.load_model('model.h5', compile=False)

optimizer=create_RMSProp_op(0.001, 0.9, 1e-07)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

total_iterations = 1000
for iteration in range(total_iterations):

    cost = train_step(X, Y_oh)

    if (iteration + 1) % 100 == 0:
        print(f'Cost after {iteration + 1} iterations: {cost}')

Y_pred_oh = model(X[:100])
Y_pred = np.argmax(Y_pred_oh, axis=1)

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(str(Y_pred[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()

ubuntu@ml:~/optimization$ ./8-main.py 
Cost after 100 iterations: 0.2860110104084015
Cost after 200 iterations: 0.18519501388072968
Cost after 300 iterations: 0.1329430341720581
Cost after 400 iterations: 0.0884409174323082
Cost after 500 iterations: 0.05956224724650383
Cost after 600 iterations: 0.04401925951242447
Cost after 700 iterations: 0.030383272096514702
Cost after 800 iterations: 0.024241114035248756
Cost after 900 iterations: 0.015456250868737698
Cost after 1000 iterations: 0.0107852378860116Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:8-RMSProp.pyHelp×Students who are done with "8. RMSProp Upgraded"Review your work×Correction of "8. RMSProp Upgraded"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

9. AdammandatoryWrite the functiondef update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):that updates a variable in place using the Adam optimization algorithm:alphais the learning ratebeta1is the weight used for the first momentbeta2is the weight used for the second momentepsilonis a small number to avoid division by zerovaris anumpy.ndarraycontaining the variable to be updatedgradis anumpy.ndarraycontaining the gradient ofvarvis the previous first moment ofvarsis the previous second moment ofvartis the time step used for bias correctionReturns: the updated variable, the new first moment, and the new second moment, respectivelyubuntu@ml:~/optimization$ cat 9-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_Adam = __import__('9-Adam').update_variables_Adam

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev1 = np.zeros((nx, 1))
    db_prev1 = 0
    dW_prev2 = np.zeros((nx, 1))
    db_prev2 = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev1, dW_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, W, dW, dW_prev1, dW_prev2, i + 1)
        b, db_prev1, db_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, b, db, db_prev1, db_prev2, i + 1)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A >= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@ml:~/optimization$ ./9-main.py
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 1.5950468370180395
Cost after 200 iterations: 0.390276184856453
Cost after 300 iterations: 0.1373790862761434
Cost after 400 iterations: 0.06963385247882237
Cost after 500 iterations: 0.043186805401891
Cost after 600 iterations: 0.029615890163981955
Cost after 700 iterations: 0.02135952185721115
Cost after 800 iterations: 0.01576513402620876
Cost after 900 iterations: 0.011813533123333355
Cost after 1000 iterations: 0.008996494409788116Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:9-Adam.pyHelp×Students who are done with "9. Adam"Review your work×Correction of "9. Adam"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

10. Adam UpgradedmandatoryWrite the functiondef create_Adam_op(alpha, beta1, beta2, epsilon):that sets up the Adam optimization algorithm inTensorFlow:alphais the learning ratebeta1is the weight used for the first momentbeta2is the weight used for the second momentepsilonis a small number to avoid division by zeroReturns:optimizerubuntu@ml:~/optimization$ cat 10-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_Adam_op = __import__('10-Adam').create_Adam_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh=one_hot(Y,10)

model = tf.keras.models.load_model('model.h5', compile=False)

optimizer=create_Adam_op(0.001, 0.9, 0.999, 1e-7)

# Training function
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

total_iterations = 1000
for iteration in range(total_iterations):

    cost = train_step(X, Y_oh)

    if (iteration + 1) % 100 == 0:
        print(f'Cost after {iteration + 1} iterations: {cost}')

Y_pred_oh = model(X[:100])
Y_pred = np.argmax(Y_pred_oh, axis=1)

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(str(Y_pred[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()

ubuntu@ml:~/optimization$ ./10-main.py 
Cost after 100 iterations: 0.19267748296260834
Cost after 200 iterations: 0.09857293218374252
Cost after 300 iterations: 0.04979228600859642
Cost after 400 iterations: 0.024284912273287773
Cost after 500 iterations: 0.01177093107253313
Cost after 600 iterations: 0.006168880499899387
Cost after 700 iterations: 0.0036108368076384068
Cost after 800 iterations: 0.0023372594732791185
Cost after 900 iterations: 0.001632177154533565
Cost after 1000 iterations: 0.001202528947032988Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:10-Adam.pyHelp×Students who are done with "10. Adam Upgraded"Review your work×Correction of "10. Adam Upgraded"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

11. Learning Rate DecaymandatoryWrite the functiondef learning_rate_decay(alpha, decay_rate, global_step, decay_step):that updates the learning rate using inverse time decay innumpy:alphais the original learning ratedecay_rateis the weight used to determine the rate at whichalphawill decayglobal_stepis the number of passes of gradient descent that have elapseddecay_stepis the number of passes of gradient descent that should occur before alpha is decayed furtherthe learning rate decay should occur in a stepwise fashionReturns: the updated value foralphaubuntu@ml:~/optimization$ cat 11-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    alpha_init = 0.1
    for i in range(100):
        alpha = learning_rate_decay(alpha_init, 1, i, 10)
        print(alpha)
ubuntu@ml:~/optimization$ ./11-main.py
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:11-learning_rate_decay.pyHelp×Students who are done with "11. Learning Rate Decay"Review your work×Correction of "11. Learning Rate Decay"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

12. Learning Rate Decay UpgradedmandatoryWrite the functiondef learning_rate_decay(alpha, decay_rate, decay_step):that creates a learning rate decay operation intensorflowusing inverse time decay:alphais the original learning ratedecay_rateis the weight used to determine the rate at whichalphawill decaydecay_stepis the number of passes of gradient descent that should occur before alpha is decayed furtherthe learning rate decay should occur in a stepwise fashionReturns: the learning rate decay operationubuntu@ml:~/optimization$ cat 12-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh=one_hot(Y,10)

model = tf.keras.models.load_model('model.h5', compile=False)

alpha = 0.1
alpha_schedule =learning_rate_decay(alpha, 1, 10)
optimizer= tf.keras.optimizers.SGD(learning_rate=alpha_schedule)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

total_iterations = 100
for iteration in range(total_iterations):

    current_learning_rate = alpha_schedule(iteration).numpy()
    print(current_learning_rate)
    cost = train_step(X, Y_oh)

ubuntu@ml:~/optimization$ ./12-main.py
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:12-learning_rate_decay.pyHelp×Students who are done with "12. Learning Rate Decay Upgraded"Review your work×Correction of "12. Learning Rate Decay Upgraded"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

13. Batch NormalizationmandatoryWrite the functiondef batch_norm(Z, gamma, beta, epsilon):that normalizes an unactivated output of a neural network using batch normalization:Zis anumpy.ndarrayof shape(m, n)that should be normalizedmis the number of data pointsnis the number of features inZgammais anumpy.ndarrayof shape(1, n)containing the scales used for batch normalizationbetais anumpy.ndarrayof shape(1, n)containing the offsets used for batch normalizationepsilonis a small number used to avoid division by zeroReturns: the normalizedZmatrixubuntu@ml:~/optimization$ cat 13-main.py 
#!/usr/bin/env python3

import numpy as np
batch_norm = __import__('13-batch_norm').batch_norm

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    Z = np.concatenate((a, b, c), axis=1)
    gamma = np.random.rand(1, 3)
    beta = np.random.rand(1, 3)
    print(Z[:10])
    Z_norm = batch_norm(Z, gamma, beta, 1e-7)
    print(Z_norm[:10])
ubuntu@ml:~/optimization$ ./13-main.py 
[[  3.52810469   3.8831507   -6.69181838]
 [  0.80031442   0.65224094  -5.39379178]
 [  1.95747597   0.729515     7.99659596]
 [  4.4817864    2.96939671   3.55263731]
 [  3.73511598   0.82687659   3.40131526]
 [ -1.95455576   3.94362119 -19.16956044]
 [  1.90017684   1.58638102  -3.24326124]
 [ -0.30271442   1.25254519 -10.38030909]
 [ -0.2064377    3.92294203  -0.20075401]
 [  0.821197     3.48051479  -3.9815039 ]]
[[ 1.48744674  0.95227432  0.82862045]
 [ 0.63640336 -0.291899    0.83717117]
 [ 0.99742624 -0.26214196  0.92538004]
 [ 1.78498593  0.6004018   0.89610557]
 [ 1.5520322  -0.22464952  0.89510874]
 [-0.22308868  0.97556057  0.74642362]
 [ 0.97954948  0.06782388  0.85133774]
 [ 0.29226936 -0.06073113  0.8043226 ]
 [ 0.32230674  0.96759734  0.87138019]
 [ 0.64291852  0.79722546  0.84647459]]
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:13-batch_norm.pyHelp×Students who are done with "13. Batch Normalization"Review your work×Correction of "13. Batch Normalization"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

14. Batch Normalization UpgradedmandatoryWrite the functiondef create_batch_norm_layer(prev, n, activation):that creates a batch normalization layer for a neural network intensorflow:previs the activated output of the previous layernis the number of nodes in the layer to be createdactivationis the activation function that should be used on the output of the layeryou should use thetf.keras.layers.Denselayer as the base layer with kernal initializertf.keras.initializers.VarianceScaling(mode='fan_avg')your layer should incorporate two trainable parameters,gammaandbeta, initialized as vectors of1and0respectivelyyou should use anepsilonof1e-7Returns: a tensor of the activated output for the layerubuntu@ml:~/optimization$ cat 14-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

lib= np.load('MNIST.npz')
X_3D = lib['X_train']
X = X_3D.reshape((X_3D.shape[0], -1))

a = create_batch_norm_layer(X, 256, tf.nn.tanh)
print(a)

ubuntu@ml:~/optimization$ ./14-main.py 
tf.Tensor(
[[ 0.23253557 -0.338765   -0.49043232 ... -0.9721543   0.5131888
   0.94621533]
 [ 0.93405616 -0.05965466 -0.04494033 ... -0.4620138   0.82534444
   0.85449564]
 [-0.01708883 -0.31570387  0.93290466 ...  0.9038917  -0.665654
  -0.3190535 ]
 ...
 [ 0.14316253  0.93222773 -0.20546891 ...  0.04174704  0.8317966
  -0.7670826 ]
 [-0.61493874 -0.45376715 -0.08518191 ... -0.6400074   0.06867824
   0.5038617 ]
 [ 0.46984434  0.852962   -0.83436054 ...  0.8256435   0.04898308
  -0.7973691 ]], shape=(50000, 256), dtype=float32)
ubuntu@ml:~/optimization$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationFile:14-batch_norm.pyHelp×Students who are done with "14. Batch Normalization Upgraded"Review your work×Correction of "14. Batch Normalization Upgraded"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

15. If you can't explain it simply, you don't understand it well enoughmandatoryWrite a blog post explaining the mechanics, pros, and cons of the following optimization techniques:Feature ScalingBatch normalizationMini-batch gradient descentGradient descent with momentumRMSProp optimizationAdam optimizationLearning rate decayYour posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.When done, please add all URLs below (blog post, tweet, etc.)Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.Add URLs here:SaveRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/optimizationHelp×Students who are done with "15. If you can't explain it simply, you don't understand it well enough"0/29pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Optimization.md`
