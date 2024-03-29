
from sklearn.model_selection import KFold
from data import load_data
from pos_tagging import RecurrentSigmoidPerceptron

def main():
    file_path = 'train.jsonl'
    data = load_data(file_path)

    epochs = 10
    learning_rate = 0.001

    kf = KFold(n_splits=5)
    fold = 0

    for train_index, test_index in kf.split(data):
        fold += 1
        print(f"Fold {fold}")

        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        input_size = 11
        output_size = 1
        
        model = RecurrentSigmoidPerceptron(input_size, output_size, learning_rate)

        model.train(train_data, epochs)
    
    file = f'./model/model_{epochs}_epoch_{learning_rate}_learning_rate'
    model.save_model(file)
    model = model.load_model(file)
    
    test_data = load_data('test.jsonl')
    model.predict(test_data)


if __name__ == "__main__":
    main()