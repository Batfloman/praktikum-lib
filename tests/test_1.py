from batfloman_praktikum_lib.io.formatters.helpers import get_first_digit_position
from batfloman_praktikum_lib.structs.measurement import Measurement

from batfloman_praktikum_lib.io.formatters import custom_format

import random

def random_exp(min_exp=-12, max_exp=12):
    e = random.randint(min_exp, max_exp)
    m = random.uniform(1, 10)
    return m * 10**e

def test_cases():
    a = Measurement(6407323005526, 640733)
    a = Measurement(37792222, 4) * 1e-7

def test_vals():
    print("\n")
    # x = Measurement(2942, 161) * 1e-3
    # print(f"{x:pm}")

    # print(a.value, a.error)
    # print(f"{f"{a}":<20} | {f"{a:pm}":<20} | {f"{a:e}":<20} | {f"{a:e3}":<20}")
    # a = Measurement(37792222, 4) * 1e-8
    # print(a.value, a.error)
    # print(f"{f"{a}":<20} | {f"{a:pm}":<20} | {f"{a:e}":<20} | {f"{a:e3}":<20}")

    # a = Measurement(6.55078e+04, 10382.3)
    # print(f"{a}")
    # a = Measurement(6.55078e+05, 10382.3)
    # print(f"{a}")


    for x in range(-10, 10):
        val = random_exp(5, 6)
        exp = random.randint(-50, 20) / 10
        err = val * 10**exp
        a = Measurement(val, err)
        print(f"{ f"{a.value:.6g}":<11} ± {f"{a.error:.6g}":<11}\t => {a}, {a:g}")

    # a = Measurement(random_exp(), "0.00001%")
    # print(f"{f"{a}":<20} | {f"{a:pm}":<20} | {f"{a:e}":<20} | {f"{a:e3}":<20}")
    # a = Measurement(random_exp(), "6%")
    # print(f"{f"{a}":<20} | {f"{a:pm}":<20} | {f"{a:e}":<20} | {f"{a:e3}":<20}")
    # a = Measurement(random_exp(), "76%")
    # print(f"{f"{a}":<20} | {f"{a:pm}":<20} | {f"{a:e}":<20} | {f"{a:e3}":<20}")
    # a = Measurement(random_exp(), "760%")
    # print(f"{f"{a}":<20} | {f"{a:pm}":<20} | {f"{a:e}":<20} | {f"{a:e3}":<20}")
    # a = Measurement(random_exp(5, 5), "760%")
    # print(f"{f"{a}":<20} | {f"{a:pm}":<20} | {f"{a:e}":<20} | {f"{a:e3}":<20}")
    #
    # for x in range(20):
    #     val = random_exp()
    #     val = Measurement(val, random.randint(1, 50)/1000 * val)
    #
    #     val_str = f"{val:}"
    #     val_str3 = f"{val:pm}"
    #     val_str4 = f"{val:e3pm}"
    #     print(f"{val_str:<20} | {val_str3:<20} | {val_str4:<20}")
