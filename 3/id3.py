import numpy as np


def en(p):
	return np.sum(-(p/np.sum(p)) * np.log2(p/np.sum(p)))

def gain(p, s):
	en0 = en(p)
	return en(p) - np.average([en(s[0]), en(s[1])], weights=[np.sum(s[0]), np.sum(s[1])]/np.sum(p))


p = [200, 250]
h = [185, 50]
l = [15, 200]
hf = [95, 20]
hm = [90, 30]
lf = [10, 80]
lm = [5, 120]

# print(en(p))
# print(gain(p, [h, l]))
print(en(h))
print(gain(h, [hf, hm]))
print(en(hf), en(hm))

print(en(l))
print(gain(l, [lf, lm]))
print(en(lf), en(lm))