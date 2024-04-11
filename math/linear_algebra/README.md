# Holberton School - Machine Learning
## Linear Algebra
This module covers the fundamentals of Linear Algebra, essential for Machine Learning.

### General Concepts

#### **What is a vector?**
A one-dimensional array of numbers.
- **Example:** `[1, 2, 3]`

#### **What is a matrix?**
A two-dimensional array of numbers.
- **Example:** `[[1, 2, 3], [4, 5, 6]]`

#### **What is a transpose?**
Flipping a matrix over its diagonal, swapping rows and columns.
- **Example:** Transpose of `[[1, 2], [3, 4]]` is `[[1, 3], [2, 4]]`.

#### **What is the shape of a matrix?**
A tuple indicating the number of rows and columns.
- **Example:** Shape of `[[1, 2, 3], [4, 5, 6]]` is `(2, 3)`.

#### **What is an axis?**
A specific dimension for operations in a matrix. Axis 0 is rows, and axis 1 is columns.
- **Example:** Sum along axis 0 of `[[1, 2], [3, 4]]` is `[4, 6]`.

#### **What is a slice?**
Selecting a subset of elements from an array or matrix.
- **Example:** The first row of `[[1, 2, 3], [4, 5, 6]]` is `[1, 2, 3]`.

#### **What are element-wise operations?**
Operations applied individually to each element in vectors or matrices.
- **Example:** Element-wise addition of `[1, 2]` and `[3, 4]` is `[4, 6]`.

#### **How do you concatenate vectors/matrices?**
Joining vectors/matrices end-to-end.
- **Example:** Concatenating `[1, 2]` with `[3, 4]` gives `[1, 2, 3, 4]`.

#### **What is the dot product?**
Multiplying corresponding elements of vectors and summing the results.
- **Example:** Dot product of `[1, 2]` and `[3, 4]` is `11`.

#### **What is matrix multiplication?**
Multiplying two matrices by taking the dot product of rows and columns.
- **Example:** Multiplying `[[1, 2], [3, 4]]` with `[[2, 0], [1, 2]]` gives `[[4, 4], [10, 8]]`.

#### **What is Numpy?**
A Python library for numerical computing, supporting large, multi-dimensional arrays and matrices.
- **Example:** Calculate mean using NumPy: `np.mean([1, 2, 3, 4])` is `2.5`.

#### **What is parallelization?**
Dividing a task into parts executed simultaneously on multiple processors to improve performance.
- **Example:** Parallelizing matrix addition can significantly reduce computation time.

#### **What is broadcasting?**
Arithmetic operations on arrays of different shapes by "stretching" the smaller array.
- **Example:** Adding `[1, 2, 3]` to `[[1], [2], [3]]` results in `[[2, 3, 4], [3, 4, 5], [4, 5, 6]]`.

---
| Task Number | Task Name                 | Brief Description                                                           |
|-------------|---------------------------|-----------------------------------------------------------------------------|
| 0           | Slice Me Up               | Slice `arr` to obtain specific segments.                                    |
| 1           | Trim Me Down              | Extract the middle columns from a 2D matrix.                                |
| 2           | Size Me Please            | Calculate and return the shape of a matrix.                                 |
| 3           | Flip Me Over              | Return the transpose of a 2D matrix.                                        |
| 4           | Line Up                   | Add two arrays element-wise and return a new list.                          |
| 5           | Across The Planes         | Perform element-wise addition of two 2D matrices.                           |
| 6           | Howdy Partner             | Concatenate two arrays into a new list.                                     |
| 7           | Gettin’ Cozy              | Concatenate two matrices along a specified axis.                            |
| 8           | Ridin’ Bareback           | Perform matrix multiplication on two 2D matrices.                           |
| 9           | Let The Butcher Slice It  | Slice a matrix to obtain specified segments.                                |
| 10          | I’ll Use My Scale         | Calculate the shape of a `numpy.ndarray`.                                   |
| 11          | The Western Exchange      | Transpose a `numpy.ndarray`.                                                |
| 12          | Bracing The Elements      | Perform element-wise operations on `mat1` and `mat2`.                       |
| 13          | Cat's Got Your Tongue     | Concatenate two matrices along a specified axis using NumPy.                |
| 14          | Saddle Up                 | Perform matrix multiplication using NumPy.                                  |


---


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
#!/usr/bin/env python3

import numpy as np
np_slice = __import__('100-slice_like_a_ninja').np_slice

mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(np_slice(mat1, axes={1: (1, 3)}))
print(mat1)
mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                 [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
print(mat2)list of integers

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

### 5. Across The Planes
Write a function def add_matrices2D(mat1, mat2): that adds two matrices element-wise:

You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If mat1 and mat2 are not the same shape, return None

### 6. Howdy Partner
Write a function def cat_arrays(arr1, arr2): that concatenates two arrays:

You can assume that arr1 and arr2 are lists of ints/floats
You must return a new list

### 7. Gettin’ Cozy
Write a function def cat_matrices2D(mat1, mat2, axis=0): that concatenates two matrices along a specific axis:

You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If the two matrices cannot be concatenated, return None

### 8. Ridin’ Bareback
Write a function def mat_mul(mat1, mat2): that performs matrix multiplication:

You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If the two matrices cannot be multiplied, return None

### 9. Let The Butcher Slice It
Complete the following source code (found below):

#!/usr/bin/env python3
import numpy as np
matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])
mat1 =  # your code here
mat2 =  # your code here
mat3 =  # your code here
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))

mat1 should be the middle two rows of matrix
mat2 should be the middle two columns of matrix
mat3 should be the bottom-right, square, 3x3 matrix of matrix
You are not allowed to use any loops or conditional statements
Your program should be exactly 10 lines

### 10. I’ll Use My Scale
Write a function def np_shape(matrix): that calculates the shape of a numpy.ndarray:

You are not allowed to use any loops or conditional statements
You are not allowed to use try/except statements
The shape should be returned as a tuple of integers

### 11. The Western Exchange
Write a function def np_transpose(matrix): that transposes matrix:

You can assume that matrix can be interpreted as a numpy.ndarray
You are not allowed to use any loops or conditional statements
You must return a new numpy.ndarray

### 12. Bracing The Elements
Write a function def np_elementwise(mat1, mat2): that performs element-wise addition, subtraction, multiplication, and division:

You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
You are not allowed to use any loops or conditional statements
You can assume that mat1 and mat2 are never empty

### 13. Cat's Got Your Tongue
Write a function def np_cat(mat1, mat2, axis=0) that concatenates two matrices along a specific axis:

You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
You must return a new numpy.ndarray
You are not allowed to use any loops or conditional statements
You may use: import numpy as np
You can assume that mat1 and mat2 are never empty

### 14. Saddle Up

Write a function def np_matmul(mat1, mat2): that performs matrix multiplication:

You can assume that mat1 and mat2 are numpy.ndarrays
You are not allowed to use any loops or conditional statements
You may use: import numpy as np
You can assume that mat1 and mat2 are never empty

### 15. Slice Like A Ninja
Write a function def np_slice(matrix, axes={}): that slices a matrix along specific axes:

You can assume that matrix is a numpy.ndarray
You must return a new numpy.ndarray
axes is a dictionary where the key is an axis to slice along and the value is a tuple representing the slice to make along that axis
You can assume that axes represents a valid slice
Hint:https://docs.python.org/3/library/functions.html#slice
