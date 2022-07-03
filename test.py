import sys

def decode_val_from_flag(val, flag):
    assert flag in ["-s", "-i", "-f", "-b", "-n"], f"Invalid Flag '{flag}'..."
    if flag == "-s":
        return val
    if flag == "-i":
        return int(val)
    if flag == "-f":
        return float(val)
    if flag == "-b":
        return (val == "True")
    if flag == "-n":
        return None

print("\n\n", sys.argv[:5], "\n\n")
args = sys.argv[1:]
arg_dict = { args[i+1]: decode_val_from_flag(args[i+2], args[i]) for i in range(0, len(args), 3) }

for k, v in arg_dict.items():
    print("\t", k, v, type(v))

#for idx, arg in enumerate(sys.argv[1:]):
# idx = 0
# while (idx+1) < len(sys.argv):
#     flag, key, value = sys.argv[idx+1], sys.argv[idx+2], sys.argv[idx+3]
#     value = decode_val_from_flag(value, flag)
#     print(f"\t{str(idx+1).zfill(2)} {flag} {key} {value} {type(value)}")
#     idx += 3