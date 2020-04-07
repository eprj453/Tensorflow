import matplotlib.pyplot as plt
import tensorflow as tf

x_data = [[1, 1],
          [3, 1],
          [1, 8],
          [6, 5],
          [6, 6]]

plt.plot([0, 5], [4, 0])
plt.plot([0, 4], [3, 8])
plt.plot([3, 6], [8, 2])
plt.scatter(x_data[0][0],x_data[0][1], c='blue' , marker='o')
plt.scatter(x_data[1][0],x_data[1][1], c='blue' , marker='o')
plt.scatter(x_data[2][0],x_data[2][1], c='orange' , marker='s')
plt.scatter(x_data[3][0],x_data[3][1], c='green' , marker='^')
plt.scatter(x_data[4][0],x_data[4][1], c='green' , marker='^')
plt.legend(['Circle or not', 'Square or not', 'Triangle or not'])
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

