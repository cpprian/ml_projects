import numpy as np

class Layer:

    def __init__(self, x=None, y=None, file_name="") -> None:
        if file_name != "":
            self.load_weights(file_name)

        self.X = x
        self.Y = y
        self.W = []
        self.activation_function = []

    def add_layer(self, n, weight_min_value=-1, weight_max_value=1, activation_function=None):
        if weight_max_value < weight_min_value:
            temp = weight_max_value
            weight_max_value = weight_min_value
            weight_min_value = temp

        self.add_activation_function(activation_function)

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
        wage = None
        with open(file_name, "r") as f:
            lines = f.readlines()

            for line in lines:
                row = np.fromstring(line, dtype=float, sep=",\n")

                if wage is None:
                    wage = row
                    continue

                wage = np.vstack((wage, row)) 

        self.add_new_wage(wage)

    def save_weights(self, file_name):
        print("Saving weights to file: " + file_name)
        print(self.W)
        with open(file_name, "w") as f:
            for wage in self.W:
                for row in wage:
                    for col in row:
                        f.write(f"{col},")
                    f.write("\n")
                f.write("\n")

    def set_W(self, W):
        print(W)
        self.W = W
        print(self.W)

    def add_new_wage(self, new_wage):
        self.W.append(new_wage)

    def add_activation_function(self, activation_function):
        self.activation_function.append(activation_function)

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