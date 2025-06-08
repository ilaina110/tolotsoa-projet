import math
import random

def generate_training_data(num_samples):
    data = []
    for _ in range(num_samples):
        a = random.uniform(-5, 5)
        b = random.uniform(-5, 5)
        c = random.uniform(-5, 5)
        delta = b**2 - 4*a*c
        
        if delta > 0:
            class_label = 1
            x1 = (-b - math.sqrt(delta)) / (2*a)
            x2 = (-b + math.sqrt(delta)) / (2*a)
        elif delta == 0:
            class_label = 0
            x1 = -b / (2*a)
            x2 = x1
        else:
            class_label = -1
            real_part = -b / (2*a)
            imag_part = math.sqrt(-delta) / (2*a)
            x1 = real_part
            x2 = imag_part

        data.append(([a, b, c], [class_label, x1, x2]))
    return data

class NeuralNetwork:
    def __init__(self):
        self.w1 = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
        self.b1 = [random.uniform(-1, 1) for _ in range(4)]
        
        self.w2 = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
        self.b2 = [random.uniform(-1, 1) for _ in range(4)]
        
        self.w3 = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
        self.b3 = [random.uniform(-1, 1) for _ in range(4)]
        
        self.w4 = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(4)]
        self.b4 = [random.uniform(-1, 1) for _ in range(3)]
        self.learning_rate = 0.1

    def sigmoid(self, x):
        
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        h1 = []
        for j in range(4):
            s = sum(self.w1[i][j] * inputs[i] for i in range(3)) + self.b1[j]
            h1.append(self.sigmoid(s))
        
        h2 = []
        for j in range(4):
            s = sum(self.w2[i][j] * h1[i] for i in range(4)) + self.b2[j]
            h2.append(self.sigmoid(s))
        
        h3 = []
        for j in range(4):
            s = sum(self.w3[i][j] * h2[i] for i in range(4)) + self.b3[j]
            h3.append(self.sigmoid(s))
        
        outputs = []
        for j in range(3):
            s = sum(self.w4[i][j] * h3[i] for i in range(4)) + self.b4[j]
            outputs.append(self.sigmoid(s))
        
        return [inputs, h1, h2, h3, outputs]

    def train(self, data, epochs):
        for epoch in range(epochs):
            total_error = 0
            for inputs, targets in data:
                activations = self.forward(inputs)
                outputs = activations[-1]
                
                errors = [targets[i] - outputs[i] for i in range(3)]
                total_error += sum(e**2 for e in errors) / 3
                
                deltas_out = [errors[i] * self.sigmoid_derivative(outputs[i]) for i in range(3)]
                
                deltas_h3 = [0] * 4
                for j in range(4):
                    for k in range(3):
                        deltas_h3[j] += deltas_out[k] * self.w4[j][k]
                    deltas_h3[j] *= self.sigmoid_derivative(activations[3][j])
                
                deltas_h2 = [0] * 4
                for j in range(4):
                    for k in range(4):
                        deltas_h2[j] += deltas_h3[k] * self.w3[j][k]
                    deltas_h2[j] *= self.sigmoid_derivative(activations[2][j])
                
                deltas_h1 = [0] * 4
                for j in range(4):
                    for k in range(4):
                        deltas_h1[j] += deltas_h2[k] * self.w2[j][k]
                    deltas_h1[j] *= self.sigmoid_derivative(activations[1][j])
                
                for j in range(3):
                    for i in range(4):
                        self.w4[i][j] += self.learning_rate * deltas_out[j] * activations[3][i]
                    self.b4[j] += self.learning_rate * deltas_out[j]
                
                for j in range(4):
                    for i in range(4):
                        self.w3[i][j] += self.learning_rate * deltas_h3[j] * activations[2][i]
                    self.b3[j] += self.learning_rate * deltas_h3[j]
                
                for j in range(4):
                    for i in range(4):
                        self.w2[i][j] += self.learning_rate * deltas_h2[j] * activations[1][i]
                    self.b2[j] += self.learning_rate * deltas_h2[j]
                
                for j in range(4):
                    for i in range(3):
                        self.w1[i][j] += self.learning_rate * deltas_h1[j] * inputs[i]
                    self.b1[j] += self.learning_rate * deltas_h1[j]
            
            if epoch % 1000 == 0:
                print(f"Époque {epoch}, Erreur moyenne: {total_error/len(data):.4f}")

def main():
    training_data = generate_training_data(500)  
    
    nn = NeuralNetwork()
    nn.train(training_data, epochs=5000) 
    
    print("\nPrédictions sur des exemples de test:")
    test_cases = [
        [1, 2, -3],
        [1, 2, 1], 
        [1, 0, 1]   
    ]
    
    for inputs in test_cases:
        a, b, c = inputs
        outputs = nn.forward(inputs)[-1]
        class_label, x1, x2 = outputs
        class_desc = "Réelles" if class_label > 0.33 else "Double" if class_label > -0.33 else "Complexes"
        print(f"\nÉquation: {a}x² + {b}x + {c} = 0")
        print(f"Classification: {class_desc} (score: {class_label:.4f})")
        print(f"Solutions prédites: x1 = {x1:.4f}, x2 = {x2:.4f}")
        
        delta = b**2 - 4*a*c
        if delta > 0:
            true_x1 = (-b - math.sqrt(delta)) / (2*a)
            true_x2 = (-b + math.sqrt(delta)) / (2*a)
            print(f"Vraies solutions: x1 = {true_x1:.4f}, x2 = {true_x2:.4f}")
        elif delta == 0:
            true_x = -b / (2*a)
            print(f"Vraie solution: x = {true_x:.4f}")
        else:
            real_part = -b / (2*a)
            imag_part = math.sqrt(-delta) / (2*a)
            print(f"Vraies solutions: x1 = {real_part:.4f} ± {imag_part:.4f}i")

if __name__ == "__main__":
    main()