import argparse


parser = argparse.ArgumentParser(description="This is a test program")

# type can be int, str, , float, bool, etc.
# this argument is optional
parser.add_argument('-argument', type=dtype, default=x, help='help message')

# this argument is required
parser.add_argument('argument', type=dtype, default=x, help='help message')

# parse the arguments
args = parser.parse_args()

# access the arguments
variable = args.argument