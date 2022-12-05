import numpy as np
import datetime


class ConvolutionalNeuralNetwork:

    def __init__(self) -> None:
        image_sections = None

    def load_input(self, filename: str, number_of_input: int=1000) -> np.ndarray:
        with open(filename, "rb") as f:
            _ = f.read(4)
            n = int.from_bytes(f.read(4), byteorder="big")
            row = int.from_bytes(f.read(4), byteorder="big")
            col = int.from_bytes(f.read(4), byteorder="big")

            if n > number_of_input:
                n = number_of_input

            input_images = np.zeros((row, col, n), dtype=np.uint8)

            i = 0
            while (byte := f.read(row * col)):
                if i == number_of_input:
                    break

                input_images[:, :, i] = np.frombuffer(byte, dtype=np.uint8).reshape(row, col)
                i += 1

        return input_images

    def load_label(self, filename: str, number_of_labels: int=1000) -> np.ndarray:
        with open(filename, "rb") as f:
            _ = f.read(4)
            n = int.from_bytes(f.read(4), byteorder="big")
            if n > number_of_labels:
                n = number_of_labels

            labels = np.zeros((n, 1))

            i = 0
            while (byte := f.read(1)):
                if i == number_of_labels:
                    break

                labels[i, 0] = int.from_bytes(byte, byteorder="big")
                i += 1

        return labels.T

    def make_filter(self, number_of_filters: int, filter_size: int=3, weight_min_value=-1, weight_max_value=1) -> np.ndarray:
        return np.random.uniform(weight_min_value, weight_max_value, size=(number_of_filters, filter_size))
            
    def convolve(self, input_image: np.ndarray, filter_conv: np.ndarray, filter_height: int, filter_width: int, step: int = 1, padding: int = 0) -> np.ndarray:
        input_image_pad = np.pad(input_image, padding, mode='constant', constant_values=0)
        input_image_height, input_image_width = input_image_pad.shape

        output_image = np.zeros((filter_height, filter_width))

        # scan scan with filter 3x3 over the image from left to right, top to bottom
        k = 0
        for i in range(input_image_height - 3 + 1):
            for j in range(input_image_width - 3 + 1):
                region = input_image_pad[i:i * step + 3, j:j * step + 3].T.reshape(1, -1)
                output_image[k, :] = region
                k += 1

        self.image_sections = output_image

        return output_image @ filter_conv.T

    def predict(self, input_image: np.ndarray, kernel: np.ndarray, wy: np.ndarray) -> np.ndarray:
        kernel_layer = self.relu(self.convolve(input_image, kernel, filter_height=kernel.shape[0], filter_width=kernel.shape[1]).flatten())
        kernel_layer_flatten = kernel_layer.reshape(-1, 1)
        layer_output = wy @ kernel_layer_flatten
        return layer_output, kernel_layer_flatten

    def train(self, alpha: int, iter: int, input_images: np.ndarray, labels: np.ndarray, filters: np.ndarray, wy: np.ndarray):
        output_filters = filters
        output_wy = wy

        # TODO: there is more than one image in the input_images -> 3D array
        for _ in range(iter):
            kernel_layer = self.relu(self.convolve(input_images, filters, filters.shape[0], filters.shape[1]))
            kernel_layer_flatten = kernel_layer.reshape(-1, 1)
            layer_output = wy @ kernel_layer_flatten

            layer_output_delta = 2 * 1/layer_output.shape[0] * (layer_output - labels)
            kernel_layer_1_delta = wy.T @ layer_output_delta
            # FIXME: activation function
            kernel_layer_1_delta = kernel_layer_1_delta * self.relu_deriv(kernel_layer_flatten)
            kernel_layer_1_delta = kernel_layer_1_delta.reshape(kernel_layer.shape)

            layer_output_weight_delta = layer_output_delta @ kernel_layer_flatten.T
            kernel_layer_1_weight_delta = kernel_layer_1_delta.T @ self.image_sections

            output_wy -= alpha * layer_output_weight_delta
            output_filters -= alpha * kernel_layer_1_weight_delta

        return output_wy, output_filters

    def accuracy(self, f_log: str, labels: np.ndarray, input_image: np.ndarray) -> float:
        acc = 0

        layer_output, _ = self.predict(input_image, self.filters, self.wy)
        for i in range(labels.shape[1]):
            output_label = np.zeros((labels.shape[0], 1))
            output_label[np.argmax(layer_output), i] = 1
            if np.array_equal(labels[:, i], output_label):
                acc += 1

        with open(f_log, "a") as f:
            f.write(f"Accuracy: {acc/self.labels.shape[1]}  {datetime.today().strftime('%D-%M-%Y %H:%M:%S')}     Acc: {acc}  Goal: {self.goal.shape[1]}\n")

        return acc / labels.shape[1]

    def relu(self, layer: np.ndarray) -> np.ndarray:
        return np.where(layer > 0, layer, 0)

    def relu_deriv(self, layer: np.ndarray) -> np.ndarray:
            return np.greater(layer, 0).astype(int)
    