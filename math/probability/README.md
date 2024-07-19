
## Probability

### Description
Question #0What does the expressionP(A | B)represent?The probability of A and BThe probability of A or BThe probability of A and not BThe probability of A given B

Question #1What does the expressionP(A ∩ B')represent?The probability of A and BThe probability of A or BThe probability of A and not BThe probability of A given B

Question #2What does the expressionP(A ∩ B)represent?The probability of A and BThe probability of A or BThe probability of A and not BThe probability of A given B

Question #3What does the expressionP(A ∪ B)represent?The probability of A and BThe probability of A or BThe probability of A and not BThe probability of A given B

Question #4The above image displays the normal distribution of male heights. What is the mode height?5'6"5'8"5'10"6’6'2"

Question #5The above image displays the normal distribution of male heights. What is the standard deviation?1"2"4"8"

Question #6The above image displays the normal distribution of male heights. What is the variance?4"8"16"64"

Question #7The above image displays the normal distribution of male heights. If a man is 6'6", what percentile would he be in?84th percentile95th percentile97.25th percentile99.7th percentile

Question #8What type of distribution is displayed above?GaussianHypergeometricChi-SquaredPoisson

Question #9What type of distribution is displayed above?GaussianHypergeometricChi-SquaredPoisson

Question #10What is the difference between a PDF and a PMF?PDF is for discrete variables while PMF is for continuous variablesPDF is for continuous variables while PMF is for discrete variablesThere is no difference

Question #11For a given distribution, the value at the 50th percentile is always:meanmedianmodeall of the above

Question #12For a given distribution, the CDF(x) where x ∈ X:The probability that X = xThe probability that X <= xThe percentile of xThe probability that X >= x

0. Initialize PoissonmandatoryCreate a classPoissonthat represents a poisson distribution:Class contructordef __init__(self, data=None, lambtha=1.):datais a list of the data to be used to estimate the distributionlambthais the expected number of occurences in a given time frameSets the instance attributelambthaSaveslambthaas a floatIfdatais not given, (i.e.None(be careful:not datahas not the same result asdata is None)):Use the givenlambthaIflambthais not a positive value or equals to 0, raise aValueErrorwith the messagelambtha must be a positive valueIfdatais given:Calculate thelambthaofdataIfdatais not alist, raise aTypeErrorwith the messagedata must be a listIfdatadoes not contain at least two data points, raise aValueErrorwith the messagedata must contain multiple valuesalexa@ubuntu-xenial:probability$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)
alexa@ubuntu-xenial:probability$ ./0-main.py 
Lambtha: 4.84
Lambtha: 5.0
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:poisson.pyHelp×Students who are done with "0. Initialize Poisson"Review your work×Correction of "0. Initialize Poisson"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

1. Poisson PMFmandatoryUpdate the classPoisson:Instance methoddef pmf(self, k):Calculates the value of the PMF for a given number of “successes”kis the number of “successes”Ifkis not an integer, convert it to an integerIfkis out of range, return0Returns the PMF value forkalexa@ubuntu-xenial:probability$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))
alexa@ubuntu-xenial:probability$ ./1-main.py 
P(9): 0.03175849616802446
P(9): 0.036265577412911795
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:poisson.pyHelp×Students who are done with "1. Poisson PMF"Review your work×Correction of "1. Poisson PMF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

2. Poisson CDFmandatoryUpdate the classPoisson:Instance methoddef cdf(self, k):Calculates the value of the CDF for a given number of “successes”kis the number of “successes”Ifkis not an integer, convert it to an integerIfkis out of range, return0Returns the CDF value forkalexa@ubuntu-xenial:probability$ cat 2-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('F(9):', p1.cdf(9))

p2 = Poisson(lambtha=5)
print('F(9):', p2.cdf(9))
alexa@ubuntu-xenial:probability$ ./2-main.py 
F(9): 0.9736102067423525
F(9): 0.9681719426208609
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:poisson.pyHelp×Students who are done with "2. Poisson CDF"Review your work×Correction of "2. Poisson CDF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

3. Initialize ExponentialmandatoryCreate a classExponentialthat represents an exponential distribution:Class contructordef __init__(self, data=None, lambtha=1.):datais a list of the data to be used to estimate the distributionlambthais the expected number of occurences in a given time frameSets the instance attributelambthaSaveslambthaas a floatIfdatais not given (i.e.None):Use the givenlambthaIflambthais not a positive value, raise aValueErrorwith the messagelambtha must be a positive valueIfdatais given:Calculate thelambthaofdataIfdatais not alist, raise aTypeErrorwith the messagedata must be a listIfdatadoes not contain at least two data points, raise aValueErrorwith the messagedata must contain multiple valuesalexa@ubuntu-xenial:probability$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('Lambtha:', e1.lambtha)

e2 = Exponential(lambtha=2)
print('Lambtha:', e2.lambtha)
alexa@ubuntu-xenial:probability$ ./3-main.py 
Lambtha: 2.1771114730906937
Lambtha: 2.0
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:exponential.pyHelp×Students who are done with "3. Initialize Exponential"Review your work×Correction of "3. Initialize Exponential"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

4. Exponential PDFmandatoryUpdate the classExponential:Instance methoddef pdf(self, x):Calculates the value of the PDF for a given time periodxis the time periodReturns the PDF value forxIfxis out of range, return0alexa@ubuntu-xenial:probability$ cat 4-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('f(1):', e1.pdf(1))

e2 = Exponential(lambtha=2)
print('f(1):', e2.pdf(1))
alexa@ubuntu-xenial:probability$ ./4-main.py 
f(1): 0.24681591903431568
f(1): 0.2706705664650693
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:exponential.pyHelp×Students who are done with "4. Exponential PDF"Review your work×Correction of "4. Exponential PDF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

5. Exponential CDFmandatoryUpdate the classExponential:Instance methoddef cdf(self, x):Calculates the value of the CDF for a given time periodxis the time periodReturns the CDF value forxIfxis out of range, return0alexa@ubuntu-xenial:probability$ cat 5-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('F(1):', e1.cdf(1))

e2 = Exponential(lambtha=2)
print('F(1):', e2.cdf(1))
alexa@ubuntu-xenial:probability$ ./5-main.py 
F(1): 0.886631473819791
F(1): 0.8646647167674654
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:exponential.pyHelp×Students who are done with "5. Exponential CDF"Review your work×Correction of "5. Exponential CDF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

6. Initialize NormalmandatoryCreate a classNormalthat represents a normal distribution:Class contructordef __init__(self, data=None, mean=0., stddev=1.):datais a list of the data to be used to estimate the distributionmeanis the mean of the distributionstddevis the standard deviation of the distributionSets the instance attributesmeanandstddevSavesmeanandstddevas floatsIfdatais not given (i.e.None(be careful:not datahas not the same result asdata is None))Use the givenmeanandstddevIfstddevis not a positive value or equals to 0, raise aValueErrorwith the messagestddev must be a positive valueIfdatais given:Calculate the mean and standard deviation ofdataIfdatais not alist, raise aTypeErrorwith the messagedata must be a listIfdatadoes not contain at least two data points, raise aValueErrorwith the messagedata must contain multiple valuesalexa@ubuntu-xenial:probability$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Mean:', n1.mean, ', Stddev:', n1.stddev)

n2 = Normal(mean=70, stddev=10)
print('Mean:', n2.mean, ', Stddev:', n2.stddev)
alexa@ubuntu-xenial:probability$ ./6-main.py 
Mean: 70.59808015534485 , Stddev: 10.078822447165797
Mean: 70.0 , Stddev: 10.0
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:normal.pyHelp×Students who are done with "6. Initialize Normal"Review your work×Correction of "6. Initialize Normal"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

7. Normalize NormalmandatoryUpdate the classNormal:Instance methoddef z_score(self, x):Calculates the z-score of a given x-valuexis the x-valueReturns the z-score ofxInstance methoddef x_value(self, z):Calculates the x-value of a given z-scorezis the z-scoreReturns the x-value ofzalexa@ubuntu-xenial:probability$ cat 7-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Z(90):', n1.z_score(90))
print('X(2):', n1.x_value(2))

n2 = Normal(mean=70, stddev=10)
print()
print('Z(90):', n2.z_score(90))
print('X(2):', n2.x_value(2))
alexa@ubuntu-xenial:probability$ ./7-main.py 
Z(90): 1.9250185174272068
X(2): 90.75572504967644

Z(90): 2.0
X(2): 90.0
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:normal.pyHelp×Students who are done with "7. Normalize Normal"Review your work×Correction of "7. Normalize Normal"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

8. Normal PDFmandatoryUpdate the classNormal:Instance methoddef pdf(self, x):Calculates the value of the PDF for a given x-valuexis the x-valueReturns the PDF value forxalexa@ubuntu-xenial:probability$ cat 8-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PSI(90):', n1.pdf(90))

n2 = Normal(mean=70, stddev=10)
print('PSI(90):', n2.pdf(90))
alexa@ubuntu-xenial:probability$ ./8-main.py 
PSI(90): 0.006206096804434349
PSI(90): 0.005399096651147344
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:normal.pyHelp×Students who are done with "8. Normal PDF"Review your work×Correction of "8. Normal PDF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

9. Normal CDFmandatoryUpdate the classNormal:Instance methoddef cdf(self, x):Calculates the value of the CDF for a given x-valuexis the x-valueReturns the CDF value forxalexa@ubuntu-xenial:probability$ cat 9-main.py 
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PHI(90):', n1.cdf(90))

n2 = Normal(mean=70, stddev=10)
print('PHI(90):', n2.cdf(90))
alexa@ubuntu-xenial:probability$ ./9-main.py 
PHI(90): 0.982902011086006
PHI(90): 0.9922398930667251
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:normal.pyHelp×Students who are done with "9. Normal CDF"Review your work×Correction of "9. Normal CDF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

10. Initialize BinomialmandatoryCreate a classBinomialthat represents a binomial distribution:Class contructordef __init__(self, data=None, n=1, p=0.5):datais a list of the data to be used to estimate the distributionnis the number of Bernoulli trialspis the probability of a “success”Sets the instance attributesnandpSavesnas an integer andpas a floatIfdatais not given (i.e.None)Use the givennandpIfnis not a positive value, raise aValueErrorwith the messagen must be a positive valueIfpis not a valid probability, raise aValueErrorwith the messagep must be greater than 0 and less than 1Ifdatais given:CalculatenandpfromdataRoundnto the nearest integer (rounded, not casting!The difference is important:int(3.7)is not the same asround(3.7))Hint: Calculatepfirst and then calculaten. Then recalculatep. Think about why you would want to do it this way?Ifdatais not alist, raise aTypeErrorwith the messagedata must be a listIfdatadoes not contain at least two data points, raise aValueErrorwith the messagedata must contain multiple valuesalexa@ubuntu-xenial:probability$ cat 10-main.py 
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('n:', b1.n, "p:", b1.p)

b2 = Binomial(n=50, p=0.6)
print('n:', b2.n, "p:", b2.p)
alexa@ubuntu-xenial:probability$ ./10-main.py 
n: 50 p: 0.606
n: 50 p: 0.6
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:binomial.pyHelp×Students who are done with "10. Initialize Binomial"Review your work×Correction of "10. Initialize Binomial"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

11. Binomial PMFmandatoryUpdate the classBinomial:Instance methoddef pmf(self, k):Calculates the value of the PMF for a given number of “successes”kis the number of “successes”Ifkis not an integer, convert it to an integerIfkis out of range, return0Returns the PMF value forkalexa@ubuntu-xenial:probability$ cat 11-main.py 
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('P(30):', b1.pmf(30))

b2 = Binomial(n=50, p=0.6)
print('P(30):', b2.pmf(30))
alexa@ubuntu-xenial:probability$ ./11-main.py 
P(30): 0.11412829839570347
P(30): 0.114558552829524
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:binomial.pyHelp×Students who are done with "11. Binomial PMF"Review your work×Correction of "11. Binomial PMF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

12. Binomial CDFmandatoryUpdate the classBinomial:Instance methoddef cdf(self, k):Calculates the value of the CDF for a given number of “successes”kis the number of “successes”Ifkis not an integer, convert it to an integerIfkis out of range, return0Returns the CDF value forkHint: use thepmfmethodalexa@ubuntu-xenial:probability$ cat 12-main.py 
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('F(30):', b1.cdf(30))

b2 = Binomial(n=50, p=0.6)
print('F(30):', b2.cdf(30))
alexa@ubuntu-xenial:probability$ ./12-main.py 
F(30): 0.5189392017296368
F(30): 0.5535236207894576
alexa@ubuntu-xenial:probability$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/probabilityFile:binomial.pyHelp×Students who are done with "12. Binomial CDF"Review your work×Correction of "12. Binomial CDF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Probability.md`
