When we pass a single linear array to plo it assumes is a list of y values and auto-generates the x
values defaulting the start to 0 but with the same length as the provided array

 ##tasks
 ###0. Line Graph
Complete the following source code to plot y as a line graph:

y should be plotted as a solid red line
The x-axis should range from 0 to 10

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # your code here
main ---
#!/usr/bin/env python3

line = __import__('0-line').line

line()
hbt-ml@Holberton-ML:~$ ./0-main.py





