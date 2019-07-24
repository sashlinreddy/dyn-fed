from fault_tolerant_ml.ml.ops.tensorpy import Tensor

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

# we want to minimize sum of squares
for i in range(100):

    x.zero_grad()

    sum_of_squares = (x * x).sum()
    sum_of_squares.backward()

    x = x - 0.1 * x.grad

    print(i, sum_of_squares)