# %%

a = [2, 4, 10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 1000, 2000, 3000]
b = [0.5, 0.6, 0.7,0.8]*4

print(len(a))

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(a, b)
plt.xlabel('Labeled Set Size')
plt.ylabel('Test Accuracy')
plt.legend(['han'])

# %%
