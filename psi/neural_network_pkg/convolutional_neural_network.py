import numpy as np
from datetime import datetime


class ConvolutionalNeuralNetwork:

    def __init__(self) -> None:
        image_sections = None
        copy_index = None

    def load_input(self, filename: str, number_of_input: int=1000) -> np.ndarray:
        with open(filename, "rb") as f:
            _ = f.read(4)
            n = int.from_bytes(f.read(4), byteorder="big")
            row = int.from_bytes(f.read(4), byteorder="big")
            col = int.from_bytes(f.read(4), byteorder="big")

            if n > number_of_input:
                n = number_of_input

            input_images = np.zeros((n, row*col), dtype=np.uint8)

            i = 0
            while (byte := f.read(row * col)):
                if i == number_of_input:
                    break

                input_images[i, :] = np.frombuffer(byte, dtype=np.uint8).reshape(1, row * col)
                i += 1

        return input_images / 255

    def load_label(self, filename: str, number_of_labels: int=1000) -> np.ndarray:
        with open(filename, "rb") as f:
            _ = f.read(4)
            n = int.from_bytes(f.read(4), byteorder="big")
            if n > number_of_labels:
                n = number_of_labels

            labels = np.zeros((10, n), dtype=np.uint8)

            i = 0
            while (byte := f.read(1)):
                if i == number_of_labels:
                    break

                val = int.from_bytes(byte, byteorder="big")
                labels[val, i] = 1
                i += 1

        return labels.T

    def make_filter(self, number_of_filters: int, filter_size: int=3, weight_min_value=-1, weight_max_value=1) -> np.ndarray:
        return np.random.uniform(weight_min_value, weight_max_value, size=(number_of_filters, filter_size))
            
    def convolve(self, input_image: np.ndarray, filter_conv: np.ndarray, filter_height: int, filter_width: int, step: int = 1, padding: int = 0) -> np.ndarray:
        input_image_pad = input_image

        sections = []
        for i in range(input_image_pad.shape[1] - filter_height + 1):
            for j in range(input_image_pad.shape[2] - filter_width + 1):
                section = input_image_pad[:, i:i + filter_height, j:j + filter_width]
                section = section.reshape(-1, 1, i + filter_height - i, j + filter_width - j)
                sections.append(section)

        self.image_sections = np.concatenate(sections, axis=1)
        self.image_sections = self.image_sections.reshape(self.image_sections.shape[0] * self.image_sections.shape[1], -1)
        return self.image_sections.dot(filter_conv)

    def train(self, alpha: int, iter: int, batch_size: int, input_images: np.ndarray, labels: np.ndarray, filters: np.ndarray, wy: np.ndarray):
        output_filters = filters
        output_wy = wy

        for _ in range(iter):
            for i in range(int(len(input_images)/batch_size)):
                batch_start, batch_end = i * batch_size, (i + 1) * batch_size
                input_batch = input_images[batch_start:batch_end]
                input_batch = input_batch.reshape(input_batch.shape[0], 28, 28)
                
                layer = self.convolve(input_batch, filters, 3, 3)
                kernel_layer = layer.reshape(batch_size, -1)
                kernel_layer = self.relu(kernel_layer)
                # dropout_mask = np.random.randint(2, size=kernel_layer.shape)
                # kernel_layer *= dropout_mask
                layer_output = self.relu(np.dot(kernel_layer, wy))

                layer_output_delta = 2 * 1/layer_output.shape[0] * (layer_output - labels[batch_start:batch_end])
                kernel_layer_1_delta = wy @ layer_output_delta.T
                # kernel_layer_1_delta *= dropout_mask.T

                kernel_layer_1_delta = kernel_layer_1_delta * self.relu_deriv(kernel_layer.T)
                kernel_layer_1_delta = kernel_layer_1_delta.reshape(kernel_layer.shape)

                layer_output_weight_delta = layer_output_delta.T @ kernel_layer
                kernel_layer_1_weight_delta = kernel_layer_1_delta.reshape(layer.shape)
                kernel_layer_1_weight_delta = self.image_sections.T @ kernel_layer_1_weight_delta

                output_wy -= alpha * layer_output_weight_delta.T
                output_filters -= alpha * kernel_layer_1_weight_delta

        return output_wy, output_filters

    def predict(self, input_image: np.ndarray, kernel: np.ndarray, wy: np.ndarray) -> np.ndarray:
        kernel_layer = self.relu(self.convolve(input_image, kernel, filter_height=3, filter_width=3).reshape(-1, 1))
        return wy.T @ kernel_layer

    def accuracy(self, f_log: str, labels: np.ndarray, input_image: np.ndarray, filters: np.ndarray, wy: np.ndarray) -> float:
        acc = 0

        for i in range(len(input_image)):
            image = input_image[i:i+1]
            image = image.reshape(image.shape[0], 28, 28)

            layer_output = self.predict(image, filters, wy)
            acc += int(np.argmax(layer_output) == np.argmax(labels[i]))      

        with open(f_log, "a") as f:
            f.write(f"Accuracy: {acc/labels.shape[0]}  {datetime.today().strftime('%D-%M-%Y %H:%M:%S')}     Acc: {acc}  Goal: {labels.shape[0]}\n")

        return acc / labels.shape[0]

    def relu(self, layer: np.ndarray) -> np.ndarray:
        return np.where(layer > 0, layer, 0)

    def relu_deriv(self, layer: np.ndarray) -> np.ndarray:
            return np.greater(layer, 0).astype(int)

    def tanh(self, x):
        return np.tanh(x)

    def tanh2deriv(self, output):
        return 1 - (output ** 2)

    def softmax(self,x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)

    def pool(self, input_image: np.ndarray, filter_size: int=2, stride: int=2) -> np.ndarray:
        input_image_height, input_image_width = input_image.shape
        output_image = np.zeros((int(input_image_height/stride), int(input_image_width/stride)))
        self.copy_index = np.zeros((int(input_image_height/stride), int(input_image_width/stride)))

        for i in range(0, input_image_height, stride):
            for j in range(0, input_image_width, stride):
                output_image[int(i/stride), int(j/stride)] = np.max(input_image[i:i+filter_size, j:j+filter_size])
                self.copy_index[int(i/stride), int(j/stride)] = np.argmax(input_image[i:i+filter_size, j:j+filter_size])

        return output_image

    def pool2(self, input_image: np.ndarray, filter_size: int=2, stride: int=2) -> np.ndarray:
        input_image_height, input_image_width = input_image.shape
        output_image = np.zeros((int(input_image_height/stride), int(input_image_width/stride)))
        self.copy_index = np.zeros((int(input_image_height/stride), int(input_image_width/stride)))

        for i in range(0, input_image_height, stride):
            for j in range(0, input_image_width, stride):
                output_image[int(i/stride), int(j/stride)] = np.max(input_image[i:i+filter_size, j:j+filter_size])
                self.copy_index[int(i/stride), int(j/stride)] = np.argmax(input_image[i:i+filter_size, j:j+filter_size])

        return output_image

    def unpool(self, input_image: np.ndarray, stride: int=2) -> np.ndarray:
        output_image = np.zeros((input_image.shape[0]*2, input_image.shape[1]*2))
        temp = np.zeros((stride, stride), dtype=float)

        for i in range(input_image.shape[0]):
            for j in range(input_image.shape[1]):
                idx = self.copy_index[i, j]
                if idx <= 1:
                    temp_idx = (0, idx)
                else:
                    temp_idx = (1, idx-2)
                
                (y, x) = temp_idx
                temp[y, int(x)] = input_image[i, j]
                if (output_image[i:i+stride, j:j+stride].shape != (2, 2)):
                    break
                output_image[i:i+stride, j:j+stride] = temp
                temp = np.zeros((stride, stride), dtype=float)

        return output_image