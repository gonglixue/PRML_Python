import numpy as np;
import matplotlib.pyplot as plt;
from numpy import *;

# use built-in function in numpy to calculate polynominal
def curveFitting():
    x = np.arange(1, 17, 1)
    y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86,
                  10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
    z1 = np.polyfit(x, y, 3)  # 多项式次数为3
    p1 = np.poly1d(z1)  # ?
    print(p1)  # 打印多项式

    yvals = p1(x) # or yvals = np.polyval(z1, x)
    plot1 = plt.plot(x, y, '*', label = 'original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)
    plt.title('polyfitting')
    plt.show()
    plt.savefig('CH1_polyfit.png')

# self-defined function calculating vandermode matrix to get polynominal
def curveFitting2():
    x = [i for i in range(1,17)]
    y = [4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86,
         10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60]
    n = len(y)  # the number of elements in training set
    degree = 3   # degree of polynominal

    vander_list = []
    for i in range(0, degree+1): #row
        vander_row = []
        for j in range(0, degree+1): #column
            item = 0;
            for k in range(0, n):
                item += pow(x[k], (i+j))
            vander_row.append(item)
        vander_list.append(vander_row)

    A_matrix = mat(vander_list);

    B_matrix = mat(zeros((degree+1,1)))
    for i in range(0, degree+1):    # each row in B_matrix
        item = 0
        for j in range(0, n):       # each element in training set
            item += (pow(x[j], i) * y[j])
        B_matrix[i] = mat([[item]])

    C_matrix = A_matrix.I * B_matrix
    # print(C_matrix)
    C_list = C_matrix.A.tolist()
    C_list.reverse()

    poly_list = []
    for i in range(0, degree+1):
        poly_list.append(C_list[i][0])

    # print(poly_list)
    # construct polynominal
    p1 = np.poly1d(poly_list)
    print(p1)
    yvals = p1(x)
    plot1 = plt.plot(x, y, '*', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)
    plt.show()
    plt.savefig('CH1_polyfit2.png')





if __name__ == '__main__':
    curveFitting()
    curveFitting2()