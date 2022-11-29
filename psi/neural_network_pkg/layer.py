import numpy as np

class Layer:

    def __init__(self, x=None, y=None, file_name="") -> None:
        if file_name != "":
            self.load_weights(file_name)

        self.X = x
        self.Y = y
        self.W = []

    def add_layer(self, n, weight_min_value=-1, weight_max_value=1):
        if weight_max_value < weight_min_value:
            temp = weight_max_value
            weight_max_value = weight_min_value
            weight_min_value = temp

        if len(self.W) == 0:
            if self.X is not None:
                col = self.X.shape[0]
            else:
                raise Exception("X is None")
        else:
            col = self.W[-1].shape[0]

        new_wage = np.random.uniform(
                        weight_min_value, 
                        weight_max_value, 
                        size=(n, col))
        self.add_new_wage(new_wage)

    def load_weights(self, file_name):
        with open(file_name, "rb") as f:
            _ = f.read(4)
            n = int.from_bytes(f.read(4), byteorder="big")
            dim = []

            for _ in range(n):
                row = int.from_bytes(f.read(4), byteorder="big")
                col = int.from_bytes(f.read(4), byteorder="big")
                dim.append((row, col))

            for i in range(n):
                self.add_new_wage(np.frombuffer(f.read(dim[i][0]*dim[i][1]), dtype=np.float64))

    def save_weights(self, file_name):
        with open(file_name, "w") as f:
            f.write(int(2004).to_bytes(4, byteorder="big"))
            f.write(len(self.W).to_bytes(4, byteorder="big"))

            for w in self.W:
                f.write(w.shape[0].to_bytes(4, byteorder="big"))
                f.write(w.shape[1].to_bytes(4, byteorder="big"))

            for w in self.W:
                f.write(w.tobytes())

    def add_new_wage(self, new_wage):
        self.W.append(new_wage)

    def load_input_and_goal(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

            for line in lines:
                row = np.fromstring(line, dtype=float, sep=" \t\n")

                x = row[:3]
                y = row[3:]

                if self.X is None:
                    self.X = x
                    continue

                if self.Y is None:
                    self.Y = y
                    continue

                self.X = np.vstack((self.X, x))
                self.Y = np.vstack((self.Y, y))

    def transpose_input(self):
        self.X = self.X.T

    def transpose_goal(self):
        self.Y = self.Y.T

    def load_input(self, filename, number_of_inputs=10):
        with open(filename, "rb") as f:
            _ = f.read(4)
            n = int.from_bytes(f.read(4), byteorder="big")
            if n > number_of_inputs:
                n = number_of_inputs

            row = int.from_bytes(f.read(4), byteorder="big")
            col = int.from_bytes(f.read(4), byteorder="big")

            first_image = np.frombuffer(f.read(row*col), dtype=np.uint8)

            self.X = np.zeros((row*col, n), dtype=np.uint8)
            self.X[:, 0] = first_image

            i = 1
            while (byte := f.read(row * col)):
                if i == number_of_inputs:
                    break
                self.X[:, i] = np.frombuffer(byte, dtype=np.uint8)
                i += 1


    def load_label(self, filename, number_of_labels=10):
        with open(filename, "rb") as f:
            _ = f.read(4)
            n = int.from_bytes(f.read(4), byteorder="big")
            if n > number_of_labels:
                n = number_of_labels

            self.Y = np.zeros((n, 1))
            self.Y[0, 0] = int.from_bytes(f.read(1), byteorder="big")

            i = 1
            while (byte := f.read(1)):
                if i == number_of_labels:
                    break

                self.Y[i, 0] = int.from_bytes(byte, byteorder="big")
                i += 1