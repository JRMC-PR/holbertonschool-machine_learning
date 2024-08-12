
## Hyperparameter Tuning

### Description
0. Initialize Gaussian ProcessmandatoryCreate the classGaussianProcessthat represents a noiseless 1D Gaussian process:Class constructor:def __init__(self, X_init, Y_init, l=1, sigma_f=1):X_initis anumpy.ndarrayof shape(t, 1)representing the inputs already sampled with the black-box functionY_initis anumpy.ndarrayof shape(t, 1)representing the outputs of the black-box function for each input inX_inittis the number of initial sampleslis the length parameter for the kernelsigma_fis the standard deviation given to the output of the black-box functionSets the public instance attributesX,Y,l, andsigma_fcorresponding to the respective constructor inputsSets the public instance attributeK, representing the current covariance kernel matrix for the Gaussian processPublic instance methoddef kernel(self, X1, X2):that calculates the covariance kernel matrix between two matrices:X1is anumpy.ndarrayof shape(m, 1)X2is anumpy.ndarrayof shape(n, 1)the kernel should use the Radial Basis Function (RBF)Returns: the covariance kernel matrix as anumpy.ndarrayof shape(m, n)root@alexa-ml2-1:~/hyperparameter_opt# cat 0-main.py 
#!/usr/bin/env python3

GP = __import__('0-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    print(gp.X is X_init)
    print(gp.Y is Y_init)
    print(gp.l)
    print(gp.sigma_f)
    print(gp.K.shape, gp.K)
    print(np.allclose(gp.kernel(X_init, X_init), gp.K))
root@alexa-ml2-1:~/hyperparameter_opt# ./0-main.py 
True
True
0.6
2
(2, 2) [[4.         0.13150595]
 [0.13150595 4.        ]]
True
root@alexa-ml2-1:~/hyperparameter_opt#Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hyperparameter_tuningFile:0-gp.pyHelp×Students who are done with "0. Initialize Gaussian Process"Review your work×Correction of "0. Initialize Gaussian Process"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

1. Gaussian Process PredictionmandatoryBased on0-gp.py, update the classGaussianProcess:Public instance methoddef predict(self, X_s):that predicts the mean and standard deviation of points in a Gaussian process:X_sis anumpy.ndarrayof shape(s, 1)containing all of the points whose mean and standard deviation should be calculatedsis the number of sample pointsReturns:mu, sigmamuis anumpy.ndarrayof shape(s,)containing the mean for each point inX_s, respectivelysigmais anumpy.ndarrayof shape(s,)containing the variance for each point inX_s, respectivelyroot@alexa-ml2-1:~/hyperparameter_opt# cat 1-main.py
#!/usr/bin/env python3

GP = __import__('1-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_s = np.random.uniform(-np.pi, 2*np.pi, (10, 1))
    mu, sig = gp.predict(X_s)
    print(mu.shape, mu)
    print(sig.shape, sig)
root@alexa-ml2-1:~/hyperparameter_opt# ./1-main.py
(10,) [ 0.20148983  0.93469135  0.14512328 -0.99831012  0.21779183 -0.05063668
 -0.00116747  0.03434981 -1.15092063  0.9221554 ]
(10,) [1.90890408 0.01512125 3.91606789 2.42958747 3.81083574 3.99817545
 3.99999903 3.9953012  3.05639472 0.37179608]
root@alexa-ml2-1:~/hyperparameter_opt#Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hyperparameter_tuningFile:1-gp.pyHelp×Students who are done with "1. Gaussian Process Prediction"Review your work×Correction of "1. Gaussian Process Prediction"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

2. Update Gaussian ProcessmandatoryBased on1-gp.py, update the classGaussianProcess:Public instance methoddef update(self, X_new, Y_new):that updates a Gaussian Process:X_newis anumpy.ndarrayof shape(1,)that represents the new sample pointY_newis anumpy.ndarrayof shape(1,)that represents the new sample function valueUpdates the public instance attributesX,Y, andKroot@alexa-ml2-1:~/hyperparameter_opt# cat 2-main.py
#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_new = np.random.uniform(-np.pi, 2*np.pi, 1)
    print('X_new:', X_new)
    Y_new = f(X_new)
    print('Y_new:', Y_new)
    gp.update(X_new, Y_new)
    print(gp.X.shape, gp.X)
    print(gp.Y.shape, gp.Y)
    print(gp.K.shape, gp.K)
root@alexa-ml2-1:~/hyperparameter_opt# ./2-main.py
X_new: [2.53931833]
Y_new: [1.99720866]
(3, 1) [[2.03085276]
 [3.59890832]
 [2.53931833]]
(3, 1) [[ 0.92485357]
 [-2.33925576]
 [ 1.99720866]]
(3, 3) [[4.         0.13150595 2.79327536]
 [0.13150595 4.         0.84109203]
 [2.79327536 0.84109203 4.        ]]
root@alexa-ml2-1:~/hyperparameter_opt#Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hyperparameter_tuningFile:2-gp.pyHelp×Students who are done with "2. Update Gaussian Process"Review your work×Correction of "2. Update Gaussian Process"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

3. Initialize Bayesian OptimizationmandatoryCreate the classBayesianOptimizationthat performs Bayesian optimization on a noiseless 1D Gaussian process:Class constructordef __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):fis the black-box function to be optimizedX_initis anumpy.ndarrayof shape(t, 1)representing the inputs already sampled with the black-box functionY_initis anumpy.ndarrayof shape(t, 1)representing the outputs of the black-box function for each input inX_inittis the number of initial samplesboundsis a tuple of(min, max)representing the bounds of the space in which to look for the optimal pointac_samplesis the number of samples that should be analyzed during acquisitionlis the length parameter for the kernelsigma_fis the standard deviation given to the output of the black-box functionxsiis the exploration-exploitation factor for acquisitionminimizeis abooldetermining whether optimization should be performed for minimization (True) or maximization (False)Sets the following public instance attributes:f: the black-box functiongp: an instance of the classGaussianProcessX_s: anumpy.ndarrayof shape(ac_samples, 1)containing all acquisition sample points, evenly spaced betweenminandmaxxsi: the exploration-exploitation factorminimize: aboolfor minimization versus maximizationYou may useGP = __import__('2-gp').GaussianProcessroot@alexa-ml2-1:~/hyperparameter_opt# cat 3-main.py 
#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
BO = __import__('3-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=2, sigma_f=3, xsi=0.05)
    print(bo.f is f)
    print(type(bo.gp) is GP)
    print(bo.gp.X is X_init)
    print(bo.gp.Y is Y_init)
    print(bo.gp.l)
    print(bo.gp.sigma_f)
    print(bo.X_s.shape, bo.X_s)
    print(bo.xsi)
    print(bo.minimize)
root@alexa-ml2-1:~/hyperparameter_opt# ./3-main.py 
True
True
True
True
2
3
(50, 1) [[-3.14159265]
 [-2.94925025]
 [-2.75690784]
 [-2.56456543]
 [-2.37222302]
 [-2.17988062]
 [-1.98753821]
 [-1.7951958 ]
 [-1.60285339]
 [-1.41051099]
 [-1.21816858]
 [-1.02582617]
 [-0.83348377]
 [-0.64114136]
 [-0.44879895]
 [-0.25645654]
 [-0.06411414]
 [ 0.12822827]
 [ 0.32057068]
 [ 0.51291309]
 [ 0.70525549]
 [ 0.8975979 ]
 [ 1.08994031]
 [ 1.28228272]
 [ 1.47462512]
 [ 1.66696753]
 [ 1.85930994]
 [ 2.05165235]
 [ 2.24399475]
 [ 2.43633716]
 [ 2.62867957]
 [ 2.82102197]
 [ 3.01336438]
 [ 3.20570679]
 [ 3.3980492 ]
 [ 3.5903916 ]
 [ 3.78273401]
 [ 3.97507642]
 [ 4.16741883]
 [ 4.35976123]
 [ 4.55210364]
 [ 4.74444605]
 [ 4.93678846]
 [ 5.12913086]
 [ 5.32147327]
 [ 5.51381568]
 [ 5.70615809]
 [ 5.89850049]
 [ 6.0908429 ]
 [ 6.28318531]]
0.05
True
root@alexa-ml2-1:~/hyperparameter_opt#Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hyperparameter_tuningFile:3-bayes_opt.pyHelp×Students who are done with "3. Initialize Bayesian Optimization"Review your work×Correction of "3. Initialize Bayesian Optimization"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

4. Bayesian Optimization - AcquisitionmandatoryBased on3-bayes_opt.py, update the classBayesianOptimization:Public instance methoddef acquisition(self):that calculates the next best sample location:Uses the Expected Improvement acquisition functionReturns:X_next, EIX_nextis anumpy.ndarrayof shape(1,)representing the next best sample pointEIis anumpy.ndarrayof shape(ac_samples,)containing the expected improvement of each potential sampleYou may usefrom scipy.stats import normroot@alexa-ml2-1:~/hyperparameter_opt# cat 4-main.py
#!/usr/bin/env python3

BO = __import__('4-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
    X_next, EI = bo.acquisition()

    print(EI)
    print(X_next)

    plt.scatter(X_init.reshape(-1), Y_init.reshape(-1), color='g')
    plt.plot(bo.X_s.reshape(-1), EI.reshape(-1), color='r')
    plt.axvline(x=X_next)
    plt.show()
root@alexa-ml2-1:~/hyperparameter_opt# ./4-main.py 
[6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
 6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
 6.77642379e-01 6.77642362e-01 6.77642264e-01 6.77641744e-01
 6.77639277e-01 6.77628755e-01 6.77588381e-01 6.77448973e-01
 6.77014261e-01 6.75778547e-01 6.72513223e-01 6.64262238e-01
 6.43934968e-01 5.95940851e-01 4.93763541e-01 3.15415142e-01
 1.01026267e-01 1.73225936e-03 4.29042673e-28 0.00000000e+00
 4.54945116e-13 1.14549081e-02 1.74765619e-01 3.78063126e-01
 4.19729153e-01 2.79303426e-01 7.84942221e-02 0.00000000e+00
 8.33323492e-02 3.25320033e-01 5.70580150e-01 7.20239593e-01
 7.65975535e-01 7.52693111e-01 7.24099594e-01 7.01220863e-01
 6.87941196e-01 6.81608621e-01 6.79006118e-01 6.78063616e-01
 6.77759591e-01 6.77671794e-01]
[4.55210364]Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hyperparameter_tuningFile:4-bayes_opt.pyHelp×Students who are done with "4. Bayesian Optimization - Acquisition"Review your work×Correction of "4. Bayesian Optimization - Acquisition"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

5. Bayesian OptimizationmandatoryBased on4-bayes_opt.py, update the classBayesianOptimization:Public instance methoddef optimize(self, iterations=100):that optimizes the black-box function:iterationsis the maximum number of iterations to performIf the next proposed point is one that has already been sampled, optimization should be stopped earlyReturns:X_opt, Y_optX_optis anumpy.ndarrayof shape(1,)representing the optimal pointY_optis anumpy.ndarrayof shape(1,)representing the optimal function valueroot@alexa-ml2-1:~/hyperparameter_opt# cat 5-main.py
#!/usr/bin/env python3

BO = __import__('5-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
    print('All sample inputs:', bo.gp.X)
root@alexa-ml2-1:~/hyperparameter_opt# ./5-main.py
Optimal X: [0.8975979]
Optimal Y: [-2.92478374]
All sample inputs: [[ 2.03085276]
 [ 3.59890832]
 [ 4.55210364]
 [ 5.89850049]
 [-3.14159265]
 [-0.83348377]
 [ 0.70525549]
 [-2.17988062]
 [ 3.01336438]
 [ 3.97507642]
 [ 1.28228272]
 [ 5.12913086]
 [ 0.12822827]
 [ 6.28318531]
 [-1.60285339]
 [-2.75690784]
 [-2.56456543]
 [ 0.8975979 ]
 [ 2.43633716]
 [-0.44879895]]
root@alexa-ml2-1:~/hyperparameter_opt#Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hyperparameter_tuningFile:5-bayes_opt.pyHelp×Students who are done with "5. Bayesian Optimization"Review your work×Correction of "5. Bayesian Optimization"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/4pts

6. Bayesian Optimization with GPyOptmandatoryWrite a python script that optimizes a machine learning model of your choice usingGPyOpt:Your script should optimize at least 5 different hyperparameters. E.g. learning rate, number of units in a layer, dropout rate, L2 regularization weight, batch sizeYour model should be optimized on a single satisficing metricYour model should save a checkpoint of its best iteration during each training sessionThe filename of the checkpoint should specify the values of the hyperparameters being tunedYour model should perform early stoppingBayesian optimization should run for a maximum of 30 iterationsOnce optimization has been performed, your script should plot the convergenceYour script should save a report of the optimization to the file'bayes_opt.txt'There are no restrictions on importsOnce you have finished your script, write a blog post describing your approach to this task. Your blog post should include:A description of what a Gaussian Process isA description of Bayesian OptimizationThe particular model that you chose to optimizeThe reasons you chose to focus on your specific hyperparametersThe reason you chose your satisficing matricYour reasoning behind any other approach choicesAny conclusions you made from performing this optimizationFinal thoughtsYour posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.When done, please add all URLs below (blog post, tweet, etc.)Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.Add URLs here:SaveRepo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hyperparameter_tuningFile:6-bayes_opt.pyHelp×Students who are done with "6. Bayesian Optimization with GPyOpt"0/19pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Hyperparameter_Tuning.md`
