import numpy

a = numpy.random.randint(0, 10, size=[4, 5, 6])
print(a)
# print(a[2:4, :])
# array([[6, 4, 1, 2, 7],
#        [4, 9, 3, 5, 9]])
# 第i列到第j列：
# print(a[:, 2:4])
# array([[4, 9],
#        [6, 4],
#        [1, 2],
#        [3, 5]])

# print(a[:, :-1])
# print(list(range(a.shape[2])))
print(range(a.shape[2]))
