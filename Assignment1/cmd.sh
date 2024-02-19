# Training
python3 src/training.py -d ./data.csv -e 2000 -l 0.1 -m 0.9 -t 0.4 -s './artifacts'

# Cross Validation
python3 src/crossval_exp.py

# Prediction
python3 src/prediction.py -l ./artifacts/best_model.pkl

# Demo
streamlit run src/demo.py