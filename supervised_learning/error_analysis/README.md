
# Error Analysis


## General Concepts

### Confusion Matrix
The **confusion matrix** is a table used to describe the performance of a classification model on a set of test data for which the true values are known. It typically contains four outcomes produced by binary classification:

- True positives (TP): Correctly predicted positive observations
- True negatives (TN): Correctly predicted negative observations
- False positives (FP): Incorrectly predicted as positive (Type I error)
- False negatives (FN): Incorrectly predicted as negative (Type II error)

### Type I and Type II Errors
- **Type I Error**: Occurs when the model incorrectly predicts the positive class (also known as a "false positive").
- **Type II Error**: Occurs when the model incorrectly predicts the negative class (also known as a "false negative").

### Sensitivity, Specificity, Precision, and Recall
- **Sensitivity (Recall or True Positive Rate)**: Measures the proportion of actual positives that are correctly identified.
  \[ Sensitivity = \frac{TP}{TP + FN} \]
- **Specificity (True Negative Rate)**: Measures the proportion of actual negatives that are correctly identified.
  \[ Specificity = \frac{TN}{TN + FP} \]
- **Precision**: Measures the proportion of positive identifications that were actually correct.
  \[ Precision = \frac{TP}{TP + FP} \]
- **Recall**: Same as sensitivity.

### F1 Score
The **F1 Score** is the harmonic mean of precision and recall. It is used as a way to combine both precision and recall into a single measure that captures both properties.
\[ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} \]

### Bias, Variance, and Irreducible Error
- **Bias**: Error introduced by approximating a real-world problem (which may be extremely complicated) with a simplified model.
- **Variance**: Error introduced by sensitivity to small fluctuations in the training set.
- **Irreducible Error**: Error that cannot be reduced by any improvement in the model. This is often due to noise in the problem itself.

### Bayes Error
**Bayes Error** represents the lowest possible error rate for any classifier of a random outcome (the noise level of the data) and is analogous to the irreducible error.

### Approximating Bayes Error
Bayes error can be approximated by increasing the dataset size and complexity of the model, then observing if improvements plateau or observing the error of the most sophisticated model.

### Calculating Bias and Variance
- **Bias**: Calculate the difference between the average prediction of our model and the correct value that we are trying to predict.
- **Variance**: Calculate the variability of model prediction for a given data point.

### Creating a Confusion Matrix
To create a confusion matrix, use the predictions from the model to compare against the actual observed outcomes and tally counts into a matrix format according to the definitions of TP, TN, FP, and FN.

### Description
0. Create ConfusionmandatoryWrite the functiondef create_confusion_matrix(labels, logits):that creates a confusion matrix:labelsis a one-hotnumpy.ndarrayof shape(m, classes)containing the correct labels for each data pointmis the number of data pointsclassesis the number of classeslogitsis a one-hotnumpy.ndarrayof shape(m, classes)containing the predicted labelsReturns: a confusionnumpy.ndarrayof shape(classes, classes)with row indices representing the correct labels and column indices representing the predicted labelsTo accompany the following main file, you are provided withlabels_logits.npz. This file does not need to be pushed to GitHub, nor will it be used to check your code.alexa@ubuntu-xenial:error_analysis$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':
    lib = np.load('labels_logits.npz')
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels, logits)
    print(confusion)
    np.savez_compressed('confusion.npz', confusion=confusion)
alexa@ubuntu-xenial:error_analysis$ ./0-main.py
[[4701.    0.   36.   17.   12.   81.   38.   11.   35.    1.]
 [   0. 5494.   36.   21.    3.   38.    7.   13.   59.    7.]
 [  64.   93. 4188.  103.  108.   17.  162.   80.  132.   21.]
 [  30.   48.  171. 4310.    2.  252.   22.   86.  128.   52.]
 [  17.   27.   35.    0. 4338.   11.   84.    9.   27.  311.]
 [  89.   57.   45.  235.   70. 3631.  123.   33.  163.   60.]
 [  47.   32.   87.    1.   64.   83. 4607.    0.   29.    1.]
 [  26.   95.   75.    7.   58.   18.    1. 4682.   13.  200.]
 [  31.  153.   82.  174.   27.  179.   64.    7. 4003.  122.]
 [  48.   37.   39.   71.  220.   49.    8.  244.   46. 4226.]]
alexa@ubuntu-xenial:error_analysis$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/error_analysisFile:0-create_confusion.pyHelp×Students who are done with "0. Create Confusion"Review your work×Correction of "0. Create Confusion"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

1. SensitivitymandatoryWrite the functiondef sensitivity(confusion):that calculates the sensitivity for each class in a confusion matrix:confusionis a confusionnumpy.ndarrayof shape(classes, classes)where row indices represent the correct labels and column indices represent the predicted labelsclassesis the number of classesReturns: anumpy.ndarrayof shape(classes,)containing the sensitivity of each classalexa@ubuntu-xenial:error_analysis$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(sensitivity(confusion))
alexa@ubuntu-xenial:error_analysis$ ./1-main.py
[0.95316302 0.96759422 0.84299517 0.84493237 0.89277629 0.80581447
 0.93051909 0.9047343  0.82672449 0.84723336]
alexa@ubuntu-xenial:error_analysis$confusion.npz:The file is coming from the output0-create_confusion.pyOr you can use this one:confusion.npzRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/error_analysisFile:1-sensitivity.pyHelp×Students who are done with "1. Sensitivity"Review your work×Correction of "1. Sensitivity"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

2. PrecisionmandatoryWrite the functiondef precision(confusion):that calculates the precision for each class in a confusion matrix:confusionis a confusionnumpy.ndarrayof shape(classes, classes)where row indices represent the correct labels and column indices represent the predicted labelsclassesis the number of classesReturns: anumpy.ndarrayof shape(classes,)containing the precision of each classalexa@ubuntu-xenial:error_analysis$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
precision = __import__('2-precision').precision

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(precision(confusion))
alexa@ubuntu-xenial:error_analysis$ ./2-main.py
[0.93033841 0.91020543 0.87359199 0.87264628 0.88494492 0.83298922
 0.90050821 0.90648596 0.86364617 0.84503099]
alexa@ubuntu-xenial:error_analysis$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/error_analysisFile:2-precision.pyHelp×Students who are done with "2. Precision"Review your work×Correction of "2. Precision"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

3. SpecificitymandatoryWrite the functiondef specificity(confusion):that calculates the specificity for each class in a confusion matrix:confusionis a confusionnumpy.ndarrayof shape(classes, classes)where row indices represent the correct labels and column indices represent the predicted labelsclassesis the number of classesReturns: anumpy.ndarrayof shape(classes,)containing the specificity of each classalexa@ubuntu-xenial:error_analysis$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))
alexa@ubuntu-xenial:error_analysis$ ./3-main.py
[0.99218958 0.98777131 0.9865429  0.98599078 0.98750582 0.98399789
 0.98870119 0.98922476 0.98600469 0.98278237]
alexa@ubuntu-xenial:error_analysis$When there are more than two classes in a confusion matrix, specificity is not a useful metric as there are inherently more actual negatives than actual positives. It is much better to use sensitivity (recall) and precision.Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/error_analysisFile:3-specificity.pyHelp×Students who are done with "3. Specificity"Review your work×Correction of "3. Specificity"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

4. F1 scoremandatoryWrite the functiondef f1_score(confusion):that calculates the F1 score of a confusion matrix:confusionis a confusionnumpy.ndarrayof shape(classes, classes)where row indices represent the correct labels and column indices represent the predicted labelsclassesis the number of classesReturns: anumpy.ndarrayof shape(classes,)containing the F1 score of each classYou must usesensitivity = __import__('1-sensitivity').sensitivityandprecision = __import__('2-precision').precisioncreate previouslyalexa@ubuntu-xenial:error_analysis$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
f1_score = __import__('4-f1_score').f1_score

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(f1_score(confusion))
alexa@ubuntu-xenial:error_analysis$ ./4-main.py
[0.94161242 0.93802288 0.8580209  0.85856574 0.88884336 0.81917654
 0.91526771 0.90560928 0.8447821  0.84613074]
alexa@ubuntu-xenial:error_analysis$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/error_analysisFile:4-f1_score.pyHelp×Students who are done with "4. F1 score"Review your work×Correction of "4. F1 score"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

5. Dealing with ErrormandatoryIn the text file5-error_handling, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex.A,B,C):Scenarios:1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low VarianceApproaches:A. Train more
B. Try a different architecture
C. Get more data
D. Build a deeper network
E. Use regularization
F. NothingRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/error_analysisFile:5-error_handlingHelp×Students who are done with "5. Dealing with Error"Review your work×Correction of "5. Dealing with Error"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

6. Compare and ContrastmandatoryGiven the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file6-compare_and_contrastMost important issue:A. High Bias
B. High Variance
C. NothingRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/error_analysisFile:6-compare_and_contrastHelp×Students who are done with "6. Compare and Contrast"Review your work×Correction of "6. Compare and Contrast"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/2pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Error_Analysis.md`
