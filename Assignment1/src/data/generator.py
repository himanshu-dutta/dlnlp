import argparse
from typing import List


def check_palindrome(inp: List[int]) -> bool:
    if len(inp) == 2:
        return inp[0] == inp[1]
    if len(inp) == 1:
        return True
    return (inp[0] == inp[len(inp) - 1]) and check_palindrome(inp[1 : len(inp) - 1])


def bin_str_generator(nbits: int, curr: List[int] = []):
    if len(curr) == nbits:
        yield curr
    else:
        yield from bin_str_generator(nbits, curr + ["0"])
        yield from bin_str_generator(nbits, curr + ["1"])


def main(dest_path: str, nbits: int, header: bool) -> None:
    with open(dest_path, "w") as fp:
        if header:
            fp.write(",".join(f"x{i}" for i in range(0, nbits)) + ",y" + "\n")

        try:
            for p in bin_str_generator(nbits):
                is_pal = "1" if check_palindrome(p) else "0"
                ln = ",".join(p) + "," + is_pal + "\n"
                fp.write(ln)
        except StopIteration as e:
            print(e)
            print("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dest_path", type=str, required=True)
    parser.add_argument("-n", "--nbits", type=int, required=True)
    parser.add_argument("-e", "--header", type=bool, default=False)
    args = parser.parse_args()
    main(args.dest_path, args.nbits, args.header)
