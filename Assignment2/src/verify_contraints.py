import argparse
import pickle

"""
Problem Statement Convention:
    W: Weight given to the previous output, y_(i-1)
    W_xy: Weight given to current input, x_i
    V_xy: Weight given to previous input, v_i = x_(i-1)
    θ: Bias/Threshold term

Implementation Convention:
    W: Weights for the previous input as well as current input
        [[ V_^] [V_NN] [V_DT] [V_JJ] [V_OT] [W_NN] [W_DT] [W_JJ] [W_OT]]
          0       1       2     3       4     5      6      7      8
    B: Bias/Threshold term
        [[θ]]
    V: Weight for the previous output
        [[W]]
"""

constraints = {
    #
    "V_^ + W_NN > θ": lambda W, B, V: (W[0][0] + W[5][0] > B[0][0]),
    "V_^ + W_DT > θ": lambda W, B, V: (W[0][0] + W[6][0] > B[0][0]),
    "V_^ + W_JJ > θ": lambda W, B, V: (W[0][0] + W[7][0] > B[0][0]),
    "V_^ + W_OT > θ": lambda W, B, V: (W[0][0] + W[8][0] > B[0][0]),
    #
    "W + V_DT + W_NN < θ": lambda W, B, V: (V[0][0] + W[2][0] + W[5][0] < B[0][0]),
    "W + V_DT + W_JJ < θ": lambda W, B, V: (V[0][0] + W[2][0] + W[7][0] < B[0][0]),
    #
    "V_JJ + W_NN < θ": lambda W, B, V: (W[3][0] + W[5][0] < B[0][0]),
    "V_JJ + W_JJ < θ": lambda W, B, V: (W[3][0] + W[7][0] < B[0][0]),
    #
    "W + V_JJ + W_NN < θ": lambda W, B, V: (V[0][0] + W[3][0] + W[5][0] < B[0][0]),
    "W + V_JJ + W_JJ < θ": lambda W, B, V: (V[0][0] + W[3][0] + W[7][0] < B[0][0]),
    #
    "V_NN + W_OT > θ": lambda W, B, V: (W[1][0] + W[8][0] > B[0][0]),
    "W + V_NN + W_OT > θ": lambda W, B, V: (V[0][0] + W[1][0] + W[8][0] > B[0][0]),
    #
    "W + V_OT + W_NN > θ": lambda W, B, V: (V[0][0] + W[4][0] + W[5][0] > B[0][0]),
    "W + V_OT + W_DT > θ": lambda W, B, V: (V[0][0] + W[4][0] + W[6][0] > B[0][0]),
    "W + V_OT + W_JJ > θ": lambda W, B, V: (V[0][0] + W[4][0] + W[7][0] > B[0][0]),
    "W + V_OT + W_OT > θ": lambda W, B, V: (V[0][0] + W[4][0] + W[8][0] > B[0][0]),
}


def constraint_verifier(weight_dict: dict[str, list]):
    W = weight_dict["W"]
    B = weight_dict["B"]
    V = weight_dict["V"]

    failed_constraints = []
    num_passed = 0

    print("W: ", W)
    print("V: ", V)
    print("B: ", B)

    for constraint_nm, constraint_fn in constraints.items():
        if constraint_fn(W, B, V):
            num_passed += 1
        else:
            failed_constraints.append(constraint_nm)

    if num_passed > 0:
        print(
            f"\033[1m\033[92m✨ Passed {num_passed}/{len(constraints)} constraints.\033[0m"
        )
    else:
        print("🥲 Failed all the constraints")
    if len(failed_constraints) > 0:
        print("\033[1m\033[91m✨ Failed Constraints:\033[0m")
        for constraint_nm in failed_constraints:
            print(f"\033[1m\033[93m\t🌟 {constraint_nm}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recurrent Perceptron Constraint Verification")
    parser.add_argument("--model_ckpt_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.model_ckpt_path, "rb") as fp:
        weight_dict = pickle.load(fp)
    constraint_verifier(weight_dict)
