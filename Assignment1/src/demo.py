import streamlit as st
from model import PalindromeModel
import numpy as np


def main():
    st.title("Palindrome Prediction Demo")

    model = PalindromeModel(10, threshold=0.4, nhidden=2)
    load_path = st.text_input("Enter the path to load the model:", "artifacts/")
    model.load(load_path + "model.pkl")

    st.success("Model loaded successfully!")

    user_input = st.text_input("Enter binary string of 10 bits:")

    if user_input:
        try:
            inp_arr = np.array(
                [float(b) for b in list(user_input)], dtype=float
            ).reshape((1, 10))
            prediction = model.predict(inp_arr).item()
            result = "Palindrome" if prediction == 1 else "Not Palindrome"
            st.write(f"Input: {user_input}, Prediction: {result}")
        except ValueError:
            st.error("Invalid input. Please enter a valid binary string of 10 bits.")


if __name__ == "__main__":
    main()
