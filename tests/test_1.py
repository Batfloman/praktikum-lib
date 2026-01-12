from batfloman_praktikum_lib.structs.measurement import Measurement

from batfloman_praktikum_lib.io.formatters import custom_format

import random

def random_exp(min_exp=-12, max_exp=12):
    e = random.randint(min_exp, max_exp)
    m = random.uniform(1, 10)
    return m * 10**e

def test_vals():
    print("\n")
    # x = Measurement(2942, 161) * 1e-3
    # print(f"{x:pm}")

    for x in range(20):
        val = random_exp()
        val = Measurement(val, random.randint(1, 50)/1000 * val)

        val_str = f"{val:}"
        val_str3 = f"{val:pm}"
        val_str4 = f"{val:e3pm}"
        print(f"{val_str:<20} | {val_str3:<20} | {val_str4:<20}")
