from batfloman_praktikum_lib.structs.measurement import Measurement

from batfloman_praktikum_lib.io.formatters import custom_format

import random

def random_exp(min_exp=-12, max_exp=12):
    e = random.randint(min_exp, max_exp)
    m = random.uniform(1, 10)
    return m * 10**e

def test_vals():
    val = random_exp()
    val = Measurement(val, random.randint(1, 50)/1000 * val)

    print(f" === {val:.4g} === ")
    print(custom_format(val, "f"))
    print(custom_format(val, "e3"))
    print(custom_format(val, ".1e3"))
