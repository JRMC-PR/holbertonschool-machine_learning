
## Bayesian Probability

### Description
Question #0Bayes’ rule states thatP(A | B) = P(B | A) * P(A) / P(B)What isP(A | B)?LikelihoodMarginal probabilityPosterior probabilityPrior probability

Question #1Bayes’ rule states thatP(A | B) = P(B | A) * P(A) / P(B)What isP(B | A)?LikelihoodMarginal probabilityPosterior probabilityPrior probability

Question #2Bayes’ rule states thatP(A | B) = P(B | A) * P(A) / P(B)What isP(A)?LikelihoodMarginal probabilityPosterior probabilityPrior probability

Question #3Bayes’ rule states thatP(A | B) = P(B | A) * P(A) / P(B)What isP(B)?LikelihoodMarginal probabilityPosterior probabilityPrior probability

0. LikelihoodmandatoryYou are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials,npatients take the drug andxpatients develop severe side effects. You can assume thatxfollows a binomial distribution.Write a functiondef likelihood(x, n, P):that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:xis the number of patients that develop severe side effectsnis the total number of patients observedPis a 1Dnumpy.ndarraycontaining the various hypothetical probabilities of developing severe side effectsIfnis not a positive integer, raise aValueErrorwith the messagen must be a positive integerIfxis not an integer that is greater than or equal to0, raise aValueErrorwith the messagex must be an integer that is greater than or equal to 0Ifxis greater thann, raise aValueErrorwith the messagex cannot be greater than nIfPis not a 1Dnumpy.ndarray, raise aTypeErrorwith the messageP must be a 1D numpy.ndarrayIf any value inPis not in the range[0, 1], raise aValueErrorwith the messageAll values in P must be in the range [0, 1]Returns: a 1Dnumpy.ndarraycontaining the likelihood of obtaining the data,xandn, for each probability inP, respectivelyalexa@ubuntu-xenial:bayesian_prob$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    likelihood = __import__('0-likelihood').likelihood

    P = np.linspace(0, 1, 11) # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(likelihood(26, 130, P))
alexa@ubuntu-xenial:bayesian_prob$ ./0-main.py 
[0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
 5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
 9.54415702e-49 1.00596671e-78 0.00000000e+00]
alexa@ubuntu-xenial:bayesian_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/bayesian_probFile:0-likelihood.pyHelp×Students who are done with "0. Likelihood"Review your work×Correction of "0. Likelihood"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/10pts

1. IntersectionmandatoryBased on0-likelihood.py, write a functiondef intersection(x, n, P, Pr):that calculates the intersection of obtaining this data with the various hypothetical probabilities:xis the number of patients that develop severe side effectsnis the total number of patients observedPis a 1Dnumpy.ndarraycontaining the various hypothetical probabilities of developing severe side effectsPris a 1Dnumpy.ndarraycontaining the prior beliefs ofPIfnis not a positive integer, raise aValueErrorwith the messagen must be a positive integerIfxis not an integer that is greater than or equal to0, raise aValueErrorwith the messagex must be an integer that is greater than or equal to 0Ifxis greater thann, raise aValueErrorwith the messagex cannot be greater than nIfPis not a 1Dnumpy.ndarray, raise aTypeErrorwith the messageP must be a 1D numpy.ndarrayIfPris not anumpy.ndarraywith the same shape asP, raise aTypeErrorwith the messagePr must be a numpy.ndarray with the same shape as PIf any value inPorPris not in the range[0, 1], raise aValueErrorwith the messageAll values in {P} must be in the range [0, 1]where{P}is the incorrect variableIfPrdoes not sum to1, raise aValueErrorwith the messagePr must sum to 1Hint: usenumpy.iscloseAll exceptions should be raised in the above orderReturns: a 1Dnumpy.ndarraycontaining the intersection of obtainingxandnwith each probability inP, respectivelyalexa@ubuntu-xenial:bayesian_prob$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    intersection = __import__('1-intersection').intersection

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
alexa@ubuntu-xenial:bayesian_prob$ ./1-main.py 
[0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
 5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
 8.67650639e-50 9.14515194e-80 0.00000000e+00]
alexa@ubuntu-xenial:bayesian_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/bayesian_probFile:1-intersection.pyHelp×Students who are done with "1. Intersection"Review your work×Correction of "1. Intersection"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/14pts

2. Marginal ProbabilitymandatoryBased on1-intersection.py, write a functiondef marginal(x, n, P, Pr):that calculates the marginal probability of obtaining the data:xis the number of patients that develop severe side effectsnis the total number of patients observedPis a 1Dnumpy.ndarraycontaining the various hypothetical probabilities of patients developing severe side effectsPris a 1Dnumpy.ndarraycontaining the prior beliefs aboutPIfnis not a positive integer, raise aValueErrorwith the messagen must be a positive integerIfxis not an integer that is greater than or equal to0, raise aValueErrorwith the messagex must be an integer that is greater than or equal to 0Ifxis greater thann, raise aValueErrorwith the messagex cannot be greater than nIfPis not a 1Dnumpy.ndarray, raise aTypeErrorwith the messageP must be a 1D numpy.ndarrayIfPris not anumpy.ndarraywith the same shape asP, raise aTypeErrorwith the messagePr must be a numpy.ndarray with the same shape as PIf any value inPorPris not in the range[0, 1], raise aValueErrorwith the messageAll values in {P} must be in the range [0, 1]where{P}is the incorrect variableIfPrdoes not sum to1, raise aValueErrorwith the messagePr must sum to 1All exceptions should be raised in the above orderReturns: the marginal probability of obtainingxandnalexa@ubuntu-xenial:bayesian_prob$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    marginal = __import__('2-marginal').marginal

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
alexa@ubuntu-xenial:bayesian_prob$ ./2-main.py 
0.008229580791426582
alexa@ubuntu-xenial:bayesian_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/bayesian_probFile:2-marginal.pyHelp×Students who are done with "2. Marginal Probability"Review your work×Correction of "2. Marginal Probability"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/14pts

3. PosteriormandatoryBased on2-marginal.py, write a functiondef posterior(x, n, P, Pr):that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:xis the number of patients that develop severe side effectsnis the total number of patients observedPis a 1Dnumpy.ndarraycontaining the various hypothetical probabilities of developing severe side effectsPris a 1Dnumpy.ndarraycontaining the prior beliefs ofPIfnis not a positive integer, raise aValueErrorwith the messagen must be a positive integerIfxis not an integer that is greater than or equal to0, raise aValueErrorwith the messagex must be an integer that is greater than or equal to 0Ifxis greater thann, raise aValueErrorwith the messagex cannot be greater than nIfPis not a 1Dnumpy.ndarray, raise aTypeErrorwith the messageP must be a 1D numpy.ndarrayIfPris not anumpy.ndarraywith the same shape asP, raise aTypeErrorwith the messagePr must be a numpy.ndarray with the same shape as PIf any value inPorPris not in the range[0, 1], raise aValueErrorwith the messageAll values in {P} must be in the range [0, 1]where{P}is the incorrect variableIfPrdoes not sum to1, raise aValueErrorwith the messagePr must sum to 1All exceptions should be raised in the above orderReturns: the posterior probability of each probability inPgivenxandn, respectivelyalexa@ubuntu-xenial:bayesian_prob$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    posterior = __import__('3-posterior').posterior

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
alexa@ubuntu-xenial:bayesian_prob$ ./3-main.py 
[0.00000000e+00 2.99729127e-03 9.63044824e-01 3.39513268e-02
 6.55839819e-06 1.26359684e-11 1.20692303e-19 6.74011797e-31
 1.05430721e-47 1.11125368e-77 0.00000000e+00]
alexa@ubuntu-xenial:bayesian_prob$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/bayesian_probFile:3-posterior.pyHelp×Students who are done with "3. Posterior"Review your work×Correction of "3. Posterior"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/14pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Bayesian_Probability.md`
