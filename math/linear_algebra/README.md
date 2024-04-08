# holbertonschool-machine_learning
Linear Algebra

##Tasks
### 0. Slice Me Up
Complete the following source code (found below):
#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 =  # your code here
arr2 =  # your code here
arr3 =  # your code here
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))

arr1 should be the first two numbers of arr
arr2 should be the last five numbers of arr
arr3 should be the 2nd through 6th numbers of arr
You are not allowed to use any loops or conditional statements
Your program should be exactly 8 lines
### 1. Trim Me Down
Complete the following source code (found below):
#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
your code here

the_middle should be a 2D matrix containing the 3rd and 4th columns of matrix
You are not allowed to use any conditional statements
You are only allowed to use one for loop
Your program should be exactly 6 lines

### 2. Size Me Please
Write a function def matrix_shape(matrix): that calculates the shape of a matrix:
#!/usr/bin/env python3

matrix_shape = __import__('2-size_me_please').matrix_shape

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))

You can assume all elements in the same dimension are of the same type/shape
The shape should be returned as a list of integers

### 3. Flip Me Over
Write a function def matrix_transpose(matrix): that returns the transpose of a 2D matrix, matrix:

You must return a new matrix
You can assume that matrix is never empty
You can assume all elements in the same dimension are of the same type/shape

### 4. Line Up
Write a function def add_arrays(arr1, arr2): that adds two arrays element-wise:

You can assume that arr1 and arr2 are lists of ints/floats
You must return a new list
If arr1 and arr2 are not the same shape, return None




