
## Advanced Linear Algebra

### Description
Question #0What is the determinant of the following matrix?[[  -7,   0,   6 ][   5,  -2, -10 ][   4,   3,   2 ]]-444414-14

Question #1What is the minor of the following matrix?[[  -7,   0,   6 ][   5,  -2, -10 ][   4,   3,   2 ]][[ 26, 50, 23 ][ -18, -38, -21 ][ 12, 40, 15 ]][[ 26, 50, 23 ][ -18, -38, -21 ][ 12, 40, 14 ]][[ 26, 50, 23 ][ -18, -39, -21 ][ 12, 40, 14 ]][[ 26, 50, 23 ][ -18, -39, -21 ][ 12, 40, 15 ]]

Question #2What is the cofactor of the following matrix?[[ 6, -9, 9 ],[ 7, 5, 0 ],[ 4, 3, -8 ]][[ -40, 56, 1 ],[ -45, -84, -54 ],[ -45, 64, 93 ]][[ -40, 56, 1 ],[ -44, -84, -54 ],[ -45, 64, 93 ]][[ -40, 56, 1 ],[ -44, -84, -54 ],[ -45, 63, 93 ]][[ -40, 56, 1 ],[ -45, -84, -54 ],[ -45, 63, 93 ]]

Question #3What is the adjugate of the following matrix?[[ -4, 1, 9 ],[ -9, -8, -5 ],[ -3, 8, 10 ]][[ -40, 62, 67 ],[ 105, -13, -101 ],[ -97, 29, 41 ]][[ -40, 62, 67 ],[ 105, -14, -101 ],[ -97, 29, 41 ]][[ -40, 62, 67 ],[ 105, -13, -101 ],[ -96, 29, 41 ]][[ -40, 62, 67 ],[ 105, -14, -101 ],[ -96, 29, 41 ]]

Question #4Is the following matrix invertible? If so, what is its inverse?[[  1,  0,  1 ][  2,  1,  2 ][  1,  0, -1 ]][[ 0.5, 0,   0.5 ][    0,  1,     2  ][ 0.5,  0,  0.5 ]][[ 0.5, 0,   0.5 ][   -2,  1,     0  ][ 0.5,  0, -0.5 ]][[ 0.5, 0,   0.5 ][    2,  1,     0  ][ 0.5,  0,  0.5 ]]It is singular

Question #5Is the following matrix invertible? If so, what is its inverse?[[ 2, 1, 2 ][ 1, 0, 1 ][ 4, 1, 4 ]][[ 4, 1, 2 ][ 1, 0, 1 ][ 4, 1, 2 ]][[ 2, 1, 4 ][ 1, 0, 1 ][ 2, 1, 4 ]][[ 4, 1, 4 ][ 1, 0, 1 ][ 2, 1, 2 ]]It is singular

Question #6GivenA= [[-2, -4, 2],[-2, 1, 2],[4, 2, 5]]v= [[2], [-3], [-1]]Wherevis an eigenvector ofA, calculateA10v[[118098], [-177147], [-59049]][[2097152], [-3145728], [-1048576]][[2048], [-3072], [-1024]]None of the above

Question #7Which of the following are also eigenvalues (λ) and eigenvectors (v) ofAwhereA= [[-2, -4, 2],[-2, 1, 2],[4, 2, 5]]λ= 5;v= [[2], [1], [1]]λ= -5;v= [[-2], [-1], [1]]λ= -3;v= [[4], [-2], [3]]λ= 6;v= [[1], [6], [16]]

Question #8What is the definiteness of the following matrix:[[ -1, 2, 0 ][ 2, -5, 2 ][ 0, 2, -6 ]]Positive definitePositive semi-definiteNegative semi-definiteNegative definiteIndefinite

Question #9What is the definiteness of the following matrix:[[ 2, 2, 1 ][ 2, 1, 3 ][ 1, 3, 8 ]]Positive definitePositive semi-definiteNegative semi-definiteNegative definiteIndefinite

Question #10What is the definiteness of the following matrix:[[ 2, 1, 1 ][ 1, 2, -1 ][ 1, -1, 2 ]]Positive definitePositive semi-definiteNegative semi-definiteNegative definiteIndefinite

0. DeterminantmandatoryWrite a functiondef determinant(matrix):that calculates the determinant of a matrix:matrixis a list of lists whose determinant should be calculatedIfmatrixis not a list of lists, raise aTypeErrorwith the messagematrix must be a list of listsIfmatrixis not square, raise aValueErrorwith the messagematrix must be a square matrixThe list[[]]represents a0x0matrixReturns: the determinant ofmatrixalexa@ubuntu-xenial:advanced_linear_algebra$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./0-main.py 
1
5
-2
0
192
matrix must be a list of lists
matrix must be a square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/advanced_linear_algebraFile:0-determinant.pyHelp×Students who are done with "0. Determinant"Review your work×Correction of "0. Determinant"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/8pts

1. MinormandatoryWrite a functiondef minor(matrix):that calculates the minor matrix of a matrix:matrixis a list of lists whose minor matrix should be calculatedIfmatrixis not a list of lists, raise aTypeErrorwith the messagematrix must be a list of listsIfmatrixis not square or is empty, raise aValueErrorwith the messagematrix must be a non-empty square matrixReturns: the minor matrix ofmatrixalexa@ubuntu-xenial:advanced_linear_algebra$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    minor = __import__('1-minor').minor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(minor(mat1))
    print(minor(mat2))
    print(minor(mat3))
    print(minor(mat4))
    try:
        minor(mat5)
    except Exception as e:
        print(e)
    try:
        minor(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./1-main.py 
[[1]]
[[4, 3], [2, 1]]
[[1, 1], [1, 1]]
[[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/advanced_linear_algebraFile:1-minor.pyHelp×Students who are done with "1. Minor"Review your work×Correction of "1. Minor"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

2. CofactormandatoryWrite a functiondef cofactor(matrix):that calculates the cofactor matrix of a matrix:matrixis a list of lists whose cofactor matrix should be calculatedIfmatrixis not a list of lists, raise aTypeErrorwith the messagematrix must be a list of listsIfmatrixis not square or is empty, raise aValueErrorwith the messagematrix must be a non-empty square matrixReturns: the cofactor matrix ofmatrixalexa@ubuntu-xenial:advanced_linear_algebra$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(cofactor(mat1))
    print(cofactor(mat2))
    print(cofactor(mat3))
    print(cofactor(mat4))
    try:
        cofactor(mat5)
    except Exception as e:
        print(e)
    try:
        cofactor(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./2-main.py 
[[1]]
[[4, -3], [-2, 1]]
[[1, -1], [-1, 1]]
[[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/advanced_linear_algebraFile:2-cofactor.pyHelp×Students who are done with "2. Cofactor"Review your work×Correction of "2. Cofactor"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

3. AdjugatemandatoryWrite a functiondef adjugate(matrix):that calculates the adjugate matrix of a matrix:matrixis a list of lists whose adjugate matrix should be calculatedIfmatrixis not a list of lists, raise aTypeErrorwith the messagematrix must be a list of listsIfmatrixis not square or is empty, raise aValueErrorwith the messagematrix must be a non-empty square matrixReturns: the adjugate matrix ofmatrixalexa@ubuntu-xenial:advanced_linear_algebra$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    adjugate = __import__('3-adjugate').adjugate

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(adjugate(mat1))
    print(adjugate(mat2))
    print(adjugate(mat3))
    print(adjugate(mat4))
    try:
        adjugate(mat5)
    except Exception as e:
        print(e)
    try:
        adjugate(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./3-main.py 
[[1]]
[[4, -2], [-3, 1]]
[[1, -1], [-1, 1]]
[[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/advanced_linear_algebraFile:3-adjugate.pyHelp×Students who are done with "3. Adjugate"Review your work×Correction of "3. Adjugate"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

4. InversemandatoryWrite a functiondef inverse(matrix):that calculates the inverse of a matrix:matrixis a list of lists whose inverse should be calculatedIfmatrixis not a list of lists, raise aTypeErrorwith the messagematrix must be a list of listsIfmatrixis not square or is empty, raise aValueErrorwith the messagematrix must be a non-empty square matrixReturns: the inverse ofmatrix, orNoneifmatrixis singularalexa@ubuntu-xenial:advanced_linear_algebra$ cat 4-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    inverse = __import__('4-inverse').inverse

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(inverse(mat1))
    print(inverse(mat2))
    print(inverse(mat3))
    print(inverse(mat4))
    try:
        inverse(mat5)
    except Exception as e:
        print(e)
    try:
        inverse(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./4-main.py 
[[0.2]]
[[-2.0, 1.0], [1.5, -0.5]]
None
[[-0.0625, -0.052083333333333336, 0.24479166666666666], [0.1875, -0.17708333333333334, -0.06770833333333333], [0.0, 0.16666666666666666, -0.08333333333333333]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:advanced_linear_algebra$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/advanced_linear_algebraFile:4-inverse.pyHelp×Students who are done with "4. Inverse"Review your work×Correction of "4. Inverse"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

5. DefinitenessmandatoryWrite a functiondef definiteness(matrix):that calculates the definiteness of a matrix:matrixis anumpy.ndarrayof shape(n, n)whose definiteness should be calculatedIfmatrixis not anumpy.ndarray, raise aTypeErrorwith the messagematrix must be a numpy.ndarrayIfmatrixis not a valid matrix, returnNoneReturn: the stringPositive definite,Positive semi-definite,Negative semi-definite,Negative definite, orIndefiniteif the matrix is positive definite, positive semi-definite, negative semi-definite, negative definite of indefinite, respectivelyIfmatrixdoes not fit any of the above categories, returnNoneYou mayimport numpy as npalexa@ubuntu-xenial:advanced_linear_algebra$ cat 5-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    definiteness = __import__('5-definiteness').definiteness
    import numpy as np

    mat1 = np.array([[5, 1], [1, 1]])
    mat2 = np.array([[2, 4], [4, 8]])
    mat3 = np.array([[-1, 1], [1, -1]])
    mat4 = np.array([[-2, 4], [4, -9]])
    mat5 = np.array([[1, 2], [2, 1]])
    mat6 = np.array([])
    mat7 = np.array([[1, 2, 3], [4, 5, 6]])
    mat8 = [[1, 2], [1, 2]]

    print(definiteness(mat1))
    print(definiteness(mat2))
    print(definiteness(mat3))
    print(definiteness(mat4))
    print(definiteness(mat5))
    print(definiteness(mat6))
    print(definiteness(mat7))
    try:
        definiteness(mat8)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:advanced_linear_algebra$ ./5-main.py 
Positive definite
Positive semi-definite
Negative semi-definite
Negative definite
Indefinite
None
None
matrix must be a numpy.ndarray
alexa@ubuntu-xenial:advanced_linear_algebra$Repo:GitHub repository:holbertonschool-machine_learningDirectory:math/advanced_linear_algebraFile:5-definiteness.pyHelp×Students who are done with "5. Definiteness"Review your work×Correction of "5. Definiteness"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/9pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Advanced_Linear_Algebra.md`
