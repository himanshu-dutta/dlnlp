IMPLEMENTATION OF RECURRENT PERCEPTRON


################################################
#   Dependecies and Pre-requisites:
################################################
Python version `3.10` has been used to work on this project. Install the project dependencies using the following command

$ pip install -r requirements.txt

########################
#   Tools
########################
Following are the tools used: numpy, sklearn (for metrics and cross-validation), matplotlib, nltk, tqdm, and streamlit.


########################
#   Commands
########################

The following commands can be used to test the functionalities of this codebase:

1. Cross Validation (k=5)

$ python3 src/crossval.py --ds_path ./artifacts/data/train.jsonl --folds 5 --learning_rate 0.01 --momentum_coeff 0.006 --epochs 5


2. Training

python3 src/training.py --train_ds_path ./artifacts/data/train.jsonl --test_ds_path ./artifacts/data/test.jsonl --model_ckpt_path ./artifacts/weights.pkl --learning_rate 0.01 --momentum_coeff 0.006 --epochs 15

3. Demo

streamlit run src/demo.py


########################
#   Best Hyerparameters
########################

| Hyerparameter                    | Value     |
|----------------------------------|-----------|
| Learning Rate                    | 0.01      |
| Momentum                         | 0.006     |
| Gradient Clipping                | (-1, 1)   |
| Number of epochs                 | 15        |
