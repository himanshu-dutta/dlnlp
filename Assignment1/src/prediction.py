from model import PalindromeModel
import argparse

import numpy as np


def main(args):
    model = PalindromeModel(
        10,
        args.threshold,
        nhidden=2,
    )
    model.load(args.load_path + "/model.pkl")
    model.weights()
    print("\033[92mLoaded model for prediction...\033[0m")

    while True:
        inp = input("Enter binary string of 10bits: ")
        inp_arr = np.array([float(b) for b in list(inp)], dtype=float).reshape((1, 10))
        res = model.predict(inp_arr).item()

        print(f"\033[96m{inp}: {'Palindrome' if res == 1 else 'Not Palindrome'}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=float, default=0.4)
    parser.add_argument("-r", "--seed", type=int, default=2)
    parser.add_argument("-l", "--load_path", type=str)

    args = parser.parse_args()

    np.random.seed(args.seed)
    main(args)
