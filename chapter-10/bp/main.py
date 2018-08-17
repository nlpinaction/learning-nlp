import bp
import mnist_loader

net = bp.Network([784, 100, 10])


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
