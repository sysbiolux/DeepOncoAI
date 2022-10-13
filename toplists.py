import random
import matplotlib.pyplot as plt
import numpy as np

results = []

for rep in range(10000):

    List1 = list(range(1500)) + list(range(1600, 1700))
    List2 = list(range(1400)) + list(range(1700, 1800))
    # List1 = 1600 genes, List2 = 1500 genes, and 1400 are common
    random.shuffle(List1)
    random.shuffle(List2)

    overlap = [x for x in List1[:20] if x in List2[:20]]
    n_common = len(overlap)
    results.append(n_common)

l = range(len(results))
avg = [np.mean(results[:x]) for x in l]
f, ax = plt.subplots()
plt.plot(l, avg)
