import numpy as np;
import matplotlib.pyplot as plt;

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


if __name__ == '__main__':
    curveFitting()