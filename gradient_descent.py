import numpy as np
import matplotlib.pyplot as plt


def gradientDescent(f, df, initX, epochs, lr):
	xs = np.zeros(epochs+1)
	xs[0] = initX
	x = initX
	for i in range(1,epochs+1):
		x = x - lr*df(x)
		xs[i] = x
		print("(x,y) after step {}: ({:.5f}, {:.5f})".format(i, x, f(x)))
	return xs

if __name__ == '__main__':
	coeffs = [float(x) for x in input("Enter coefficients of target polynomial function: ").split()]

	f = np.poly1d(coeffs)
	
	learningRate = float(input("Enter learning rate: "))
	initX = float(input("Enter initial x value: "))
	epochs = int(input("Enter epochs: "))
	
	xHistory = gradientDescent(f, f.deriv(), initX, epochs, learningRate)
	
	xdiff = max(xHistory) - min(xHistory)
	xmid = (max(xHistory) + min(xHistory))/2
	t = np.arange(xmid-(xdiff/2*1.25), xmid+(xdiff/2*1.25),0.1)
	plt.figure()
	plt.subplot(211)
	plt.plot(t, f(t))
	plt.plot(xHistory, f(xHistory), 'r-')
	plt.plot(xHistory, f(xHistory), 'ro', markersize=2)
	plt.xlabel('x')
	plt.ylabel('y')

	t = np.arange(xmid-(xdiff*2), xmid+(xdiff*2),0.1)
	plt.subplot(212)
	plt.plot(t, f(t))
	plt.plot(xHistory, f(xHistory), 'r-')
	plt.plot(xHistory, f(xHistory), 'ro', markersize=2)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
