# Palindrome Neural Network

## Dependecies and Pre-requisites

Python version `3.11` has been used to work on this project.

Install the project dependencies using the following command

```sh
$ pip install -r requirements.txt
```

## Tools

Following are the tools used: `numpy`, `sklearn` (for metrics and cross-validation), `matplotlib`, `imblearn`, `tqdm`, `streamlit`, `pandas`.


## Commands

The following commands can be used to test the functionalities of this codebase:

1. Cross Validation (k=4)
```sh
$ python3 src/crossval_exp.py
```

2. Training
```sh
python3 src/training.py -d ./data.csv -e 2000 -l 0.1 -m 0.09 -t 0.4 -s './artifacts'
```

3. Prediction 
```sh
python3 src/prediction.py -l ./artifacts/best_model.pkl
```

4. Demo
```sh
streamlit run src/demo.py
```



## Best Hyerparameters

| Hyerparameter                    | Value |
|----------------------------------|-------|
| Learning Rate                    | 0.1   |
| Momentum                         | 0.09  |
| Number of Hidden Neurons         | 2     |
| Threshold for 0/1 Classification | 0.4   |
| Number of epochs                 | 2000 |
