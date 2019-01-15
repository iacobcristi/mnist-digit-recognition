import cPickle, gzip
import numpy as np

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def activation(x):
	if x > 0:
		return 1
	return 0

learningRate = 0.01

images = train_set[0]
targets = train_set[1]

weights = np.random.uniform(0,1,(10,784))

for nr in range(0,10):
	for i in range(0,50000):
		x = images[i]
		t = targets[i]
		z = np.dot(weights[nr],x)
		output = activation(z)
		if nr == t:
			target = 1
		else:
			target = 0
		adjust = np.multiply((target - output) * learningRate, x)
		weights[nr] = np.add(weights[nr], adjust)

images = test_set[0]
targets = test_set[1]

OK = 0

for i in range(0, 10000):
	vec = []
	for j in range(0,10):
		vec.append(np.dot(weights[j],images[i]))
	if np.argmax(vec) == targets[i]:
		OK = OK + 1

print("The network recognized " + str(OK) +'/'+ "10000")