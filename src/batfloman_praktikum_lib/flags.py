import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-q", "--quiet",
    action="store_true",
    help="Keine Ausgaben"
)

def check_quiet():
    args, _ = parser.parse_known_args()
    return args.quiet
