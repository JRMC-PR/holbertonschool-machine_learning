
## Multivariate Probability

### Description
Question #0px, y(x, y) =P(X = x)P(Y = y)P(X = x | Y = y)P(X = x | Y = y)P(Y = y)P(Y = y | X = x)P(Y = y | X = x)P(X = x)P(X = x ∩ Y = y)P(X = x ∪ Y = y)

Question #1Thei,jthentry in the covariance matrix isthe variance of variableiplus the variance of variablejthe variance ofiifi == jthe same as thej,ithentrythe variance of variableiand variablej

Question #2The correlation coefficient of the variables X and Y is defined as:ρ = σXY2ρ = σXYρ = σXY/ ( σXσY)ρ = σXY/ ( σXXσYY)

0. Mean and CovariancemandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef mean_cov(X):that calculates the mean and covariance of a data set:Xis anumpy.ndarrayof shape(n, d)containing the data set:nis the number of data pointsdis the number of dimensions in each data pointIfXis not a 2Dnumpy.ndarray, raise aTypeErrorwith the messageX must be a 2D numpy.ndarrayIfnis less than 2, raise aValueErrorwith the messageX must contain multiple data pointsReturns:mean,cov:meanis anumpy.ndarrayof shape(1, d)containing the mean of the data setcovis anumpy.ndarrayof shape(d, d)containing the covariance matrix of the data setYou are not allowed to use the functionnumpy.covalexa@ubuntu-xenial:multivariate_prob$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    mean_cov = __import__('0-mean_cov').mean_cov

    np.random.seed(0)
    X = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000)
    mean, cov = mean_cov(X)
    print(mean)
    print(cov)
alexa@ubuntu-xenial:multivariate_prob$ ./0-main.py 
[[12.04341828 29.92870885 10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
alexa@ubuntu-xenial:multivariate_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/multivariate_probFile:0-mean_cov.pyHelp×Students who are done with "0. Mean and Covariance"Review your work×Correction of "0. Mean and Covariance"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:README.md file exists and is not emptyFile existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNot allowed to usenumpy.covOutput check:Xis not a 2Dnumpy.ndarrayOutput check:Xcontains only one data pointOutput check:meanis calculated correctlyOutput check:covis calculated correctlyOutput check:dis 1Pycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×0. Mean and CovarianceCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---7/7pts

1. CorrelationmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef correlation(C):that calculates a correlation matrix:Cis anumpy.ndarrayof shape(d, d)containing a covariance matrixdis the number of dimensionsIfCis not anumpy.ndarray, raise aTypeErrorwith the messageC must be a numpy.ndarrayIfCdoes not have shape(d, d), raise aValueErrorwith the messageC must be a 2D square matrixReturns anumpy.ndarrayof shape(d, d)containing the correlation matrixalexa@ubuntu-xenial:multivariate_prob$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    correlation = __import__('1-correlation').correlation

    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    Co = correlation(C)
    print(C)
    print(Co)
alexa@ubuntu-xenial:multivariate_prob$ ./1-main.py 
[[ 36 -30  15]
 [-30 100 -20]
 [ 15 -20  25]]
[[ 1.  -0.5  0.5]
 [-0.5  1.  -0.4]
 [ 0.5 -0.4  1. ]]
alexa@ubuntu-xenial:multivariate_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/multivariate_probFile:1-correlation.pyHelp×Students who are done with "1. Correlation"Review your work×Correction of "1. Correlation"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npOutput check:Xis not anumpy.ndarrayOutput check:Xis not a 2D square matrixOutput check: normalOutput check:Chas shape(1, 1)Pycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×1. CorrelationCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---6/6pts

2. InitializemandatoryScore:100.00%(Checks completed: 100.00%)Create the classMultiNormalthat represents a Multivariate Normal distribution:class constructordef __init__(self, data):datais anumpy.ndarrayof shape(d, n)containing the data set:nis the number of data pointsdis the number of dimensions in each data pointIfdatais not a 2Dnumpy.ndarray, raise aTypeErrorwith the messagedata must be a 2D numpy.ndarrayIfnis less than 2, raise aValueErrorwith the messagedata must contain multiple data pointsSet the public instance variables:mean- anumpy.ndarrayof shape(d, 1)containing the mean ofdatacov- anumpy.ndarrayof shape(d, d)containing the covariance matrixdataYou are not allowed to use the functionnumpy.covalexa@ubuntu-xenial:multivariate_prob$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    print(mn.mean)
    print(mn.cov)
alexa@ubuntu-xenial:multivariate_prob$ ./2-main.py 
[[12.04341828]
 [29.92870885]
 [10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
alexa@ubuntu-xenial:multivariate_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/multivariate_probFile:multinormal.pyHelp×Students who are done with "2. Initialize"Review your work×Correction of "2. Initialize"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNot allowed to usenumpy.covOutput check:datais not a 2Dnumpy.ndarrayOutput check:datacontains only one data pointOutput check:meanis calculated correctlyOutput check:covis calculated correctlyOutput check:dis 1Pycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×2. InitializeCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---7/7pts

3. PDFmandatoryScore:100.00%(Checks completed: 100.00%)Update the classMultiNormal:public instance methoddef pdf(self, x):that calculates the PDF at a data point:xis anumpy.ndarrayof shape(d, 1)containing the data point whose PDF should be calculateddis the number of dimensions of theMultinomialinstanceIfxis not anumpy.ndarray, raise aTypeErrorwith the messagex must be a numpy.ndarrayIfxis not of shape(d, 1), raise aValueErrorwith the messagex must have the shape ({d}, 1)Returns the value of the PDFYou are not allowed to use the functionnumpy.covalexa@ubuntu-xenial:multivariate_prob$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
alexa@ubuntu-xenial:multivariate_prob$ ./3-main.py 
[[ 8.20311936]
 [32.84231319]
 [ 9.67254478]]
0.00022930236202143827
alexa@ubuntu-xenial:multivariate_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/multivariate_probFile:multinormal.pyHelp×Students who are done with "3. PDF"Review your work×Correction of "3. PDF"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import any module exceptimport numpy as npNot allowed to usenumpy.covOutput check:xis not anumpy.ndarrayOutput check:xis not of shape(d, 1)Output check: normalPycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×3. PDFCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---5/5pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Multivariate_Probability.md`
