import numpy as np


class ConvolutionalNeuralNetwork:

    def __init__(self) -> None:
        pass

    def image_section(self, image, row_from, row_to, col_from, col_to):
        section = image[row_from:row_to, col_from:col_to]
        return section.reshape(-1, row_to-row_from, col_to-col_from)

    def make_image_sections(self, image, kernel_shape):
        h_kernel, w_kernel = kernel_shape
        h_image, w_image = image.shape

        output = np.zeros((h_kernel, w_kernel, 1), dtype=np.float32)

        idx = 0
        for col in range(w_image):
            for row in range(h_image - 1):
                output[:, idx] = self.image_section(image, row, row + 2, col, col + 1)
                idx += 1

        return output.reshape(h_kernel, w_kernel)
                    
    def convolve(self, image, conv_filter, stride=1, padding=0):
        h_filter, w_filter = conv_filter.shape
        h_image, w_image = image.shape

        h_output = int((h_image - h_filter + 2 * padding) / stride) + 1
        w_output = int((w_image - w_filter + 2 * padding) / stride) + 1

        output = np.zeros((h_output, w_output))

        for r in range(h_output):
            for c in range(w_output):
                output[r, c] = np.sum(
                    self.image_section(image, r, r + h_filter, c, c + w_filter) * conv_filter)

        return output

    def train(self, alpha, image_sections, wy, kernel_layer, expected_output, kernels):
        kernel_layer_flatten = self.relu_deriv(kernel_layer.flatten()).reshape(-1, 1)

        layer_output = np.dot(wy, kernel_layer_flatten)
        layer_output_delta = 2 * 1/expected_output.shape[0] * (layer_output - expected_output)

        kernel_layer_1_delta = np.dot(wy.T, layer_output_delta)
        kernel_layer_1_delta = kernel_layer_1_delta.reshape(kernel_layer.shape)

        layer_output_weight_delta = np.dot(layer_output_delta, kernel_layer_flatten.T)
        kernel_layer_1_weight_delta = kernel_layer_1_delta.T @ image_sections

        wy -= alpha * layer_output_weight_delta
        kernels -= alpha * kernel_layer_1_weight_delta

        return wy, kernels

    def relu_deriv(self, layer):
        return layer * (layer > 0)

