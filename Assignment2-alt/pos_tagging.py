import numpy as np
from tqdm import tqdm
import pickle

class RecurrentSigmoidPerceptron:
    def __init__(self, input_size, output_size, learning_rate):
        self.W_ih = np.random.randn(input_size, 1) 
        self.learning_rate = learning_rate
        
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        return self.sigmoid(np.dot(self.W_ih.T, input_data))

    def backward(self, input_data, output, target):
        d_output = output - target
        d_W_ih = np.dot(input_data, d_output.T * self.sigmoid_derivative(output))
        self.W_ih -= self.learning_rate * d_W_ih

    def train(self, data, epochs):
        # for epoch in range(epochs):
        for epoch in tqdm(range(epochs)):
                    
            total_loss = 0
            total_correct = 0
            total_examples = 0
            # for example in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", unit="example"):
            for example in data:
                pos_tags = example["pos_tags"]
                chunk_tags = example["chunk_tags"]
                feedback_prev = np.zeros((1, 1))

                for i in range(len(pos_tags)):
                    prev_word_one_hot = np.zeros((5, 1))
                    current_word_one_hot = np.zeros((4, 1))

                    if i > 0:
                        prev_word_one_hot[pos_tags[i - 1]] = 1
                        current_word_one_hot[pos_tags[i] - 1] = 1
                    else:
                        prev_word_one_hot[0] = 1
                        current_word_one_hot[pos_tags[i] - 1] = 1

                    input_data = np.concatenate((prev_word_one_hot, current_word_one_hot, np.ones((1, 1))))
                    input_data = np.concatenate((input_data, feedback_prev), axis=0)
                    target = chunk_tags[i]

                    output = self.forward(input_data)

                    self.backward(input_data, output, target)

                    predicted_tag = 1 if output >= 0.5 else 0
                    actual_tag = target

                    if predicted_tag == actual_tag:
                        total_correct += 1

                    total_loss += (output - target) ** 2
                    total_examples += 1

                    feedback_prev = output

            accuracy = total_correct / total_examples
            avg_loss = total_loss / len(data)

            print(f"Epoch {epoch + 1}, Loss: {avg_loss}, Accuracy: {accuracy}")
            
    def predict(self, data):
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for example in data:
            pos_tags = example["pos_tags"]
            chunk_tags = example["chunk_tags"]
            feedback_prev = np.zeros((1, 1))

            for i in range(len(pos_tags)):
                prev_word_one_hot = np.zeros((5, 1))
                current_word_one_hot = np.zeros((4, 1))

                if i > 0:
                    prev_word_one_hot[pos_tags[i - 1]] = 1
                    current_word_one_hot[pos_tags[i] - 1] = 1
                else:
                    prev_word_one_hot[0] = 1
                    current_word_one_hot[pos_tags[i] - 1] = 1

                input_data = np.concatenate((prev_word_one_hot, current_word_one_hot, np.ones((1, 1))))
                input_data = np.concatenate((input_data, feedback_prev), axis=0)
                target = chunk_tags[i]

                output = self.forward(input_data)

                predicted_tag = 1 if output >= 0.5 else 0
                actual_tag = target

                if predicted_tag == actual_tag:
                    total_correct += 1

                total_loss += (output - target) ** 2  
                total_examples += 1

                feedback_prev = output

        accuracy = total_correct / total_examples
        avg_loss = total_loss / total_examples  

        print(f"Predict: Loss: {avg_loss}, Accuracy: {accuracy}")