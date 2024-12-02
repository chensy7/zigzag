def get_layer_node_input_format_conv(
    B,K,G,IX,IY,OX,OY,C,FX,FY,strides,dilations,padding,preds
):
    d = {}
    d["operator_type"] = "Conv"
    d["equation"] = "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]"
    d["loop_dim_size"] = {
        "B": B,
        "K": K,
        "G": G,
        "OX": OX,
        "OY": OY,
        "C": C,
        "FX": FX,
        "FY": FY,
    }
    d["pr_loop_dim_size"] = {"IX": IX, "IY": IY}
    d["dimension_relations"] = [
        f"ix={strides}*ox+{dilations}*fx",
        f"iy={strides}*oy+{dilations}*fy",
    ]
    d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
    # d["operand_source"] =  {'W': [], 'I': []}
    d["constant_operands"] = ["W"]

    if preds is None: 
        d["operand_source"] = {"I": []}
    else:
        d["operand_source"] = {"I": [preds]}

    d["padding"] = {
        "IY": (padding, padding),
        "IX": (padding, padding),
    }

    return d

def convert_into_list(size):
    size = size.split("[")[-1].split("]")[0].split(",")
    return [int(s) for s in size]

workload = {}
with open("voresnet.txt", "r") as f:
    lines = [line[:-1] for line in f]
    for i in range(len(lines)):
        workload[i] = get_layer_node_input_format_conv(*convert_into_list(lines[i]),i-1 if i>0 else None)

with open("voresnet.py", "w") as f:
    f.write("workload = {\n")
    for k in workload.keys():
        f.write(str(k)+":")
        f.write(str(workload[k]))
        f.write(",\n")
    f.write("}")