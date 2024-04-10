import numpy
import tqdm


class A:
    def __init__(self) -> None:
        print("A::init")


class B(A):
    def __init__(self) -> None:
        super().__init__()
        print("B::init")


a = A()
b = B()
# a = numpy.random.randint(0, 10, size=[4, 5])
# print(a)


# a1 = a[:, :-1]
# feature_index_list = list(range(a1.shape[1]))
# a2 = a[:, feature_index_list]
# print(f"a1:{a1},\n a2:{a2}")
