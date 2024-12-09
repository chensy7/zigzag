from math import ceil

def get_layer_node_input_format_conv2d(
    kernel_shape,
    strides,
    dilations,
    groups,
    padding,
    ia_shape,
    oa_shape
):
    d = {}
    d["operator_type"] = "Conv2d"
    d["equation"] = "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]"
    assert (
        ia_shape[0] == oa_shape[0]
    ), "Batch size is different for input and output activations."
    B = ia_shape[0]
    G = 1
    K = oa_shape[1]
    OX = oa_shape[3]
    OY = oa_shape[2]
    C = ia_shape[1]
    IX = ia_shape[3]
    IY = ia_shape[2]
    FX = kernel_shape[0]
    FY = kernel_shape[1]
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

    d["padding"] = {
        "IY": (padding, padding),
        "IX": (padding, padding),
    }
    macs = B*K*OX*OY*C*FX*FY
    print(f"MAC: {macs}")
    return d, macs

def get_layer_node_input_format_conv3d(
    kernel_shape,
    strides,
    dilations,
    groups,
    padding,
    ia_shape,
    oa_shape
):
    d = {}
    d["operator_type"] = "Conv3d"
    d["equation"] = "O[b][g][k][oz][oy][ox]+=W[g][k][c][fz][fy][fx]*I[b][g][c][iz][iy][ix]"
    assert (
        ia_shape[0] == oa_shape[0]
    ), "Batch size is different for input and output activations."
    B = ia_shape[0]
    G = 1
    K = oa_shape[1]
    OX = oa_shape[4]
    OY = oa_shape[3]
    OZ = oa_shape[2]
    C = ia_shape[1]
    IX = ia_shape[4]
    IY = ia_shape[3]
    IZ = ia_shape[2]
    FX = kernel_shape[0]
    FY = kernel_shape[1]
    FZ = kernel_shape[2]
    d["loop_dim_size"] = {
        "B": B,
        "K": K,
        "G": G,
        "OX": OX,
        "OY": OY,
        "OZ": OZ,
        "C": C,
        "FX": FX,
        "FY": FY,
        "FZ": FZ,
    }
    d["pr_loop_dim_size"] = {"IX": IX, "IY": IY, "IZ": IZ}
    d["dimension_relations"] = [
        f"ix={strides}*ox+{dilations}*fx",
        f"iy={strides}*oy+{dilations}*fy",
        f"iz={strides}*oz+{dilations}*fz",
    ]
    d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
    # d["operand_source"] =  {'W': [], 'I': []}
    d["constant_operands"] = ["W"]

    d["padding"] = {
        "IZ": (padding, padding, padding),
        "IY": (padding, padding, padding),
        "IX": (padding, padding, padding),
    }
    macs = B*K*OX*OY*OZ*C*FX*FY*FZ
    print(f"MAC: {macs}")

    return d, macs

def convert_into_list(size):
    size = size.split("[")[-1].split("]")[0].split(",")
    return [int(s) for s in size]

workload = {}
total_macs = 0
with open("mvs_gi.txt", "r") as f:
    lines = [line[:-1] for line in f]
    for i in range(len(lines)//3):
        ia = convert_into_list(lines[i*3])
        wt = convert_into_list(lines[i*3+1])
        oa = convert_into_list(lines[i*3+2])
        stride = 2 if oa[-1] < ia[-1] else 1
        group = 1
        padding = 1 if wt[-1] == 3 else 2
        if len(ia) == 4:
            output = get_layer_node_input_format_conv2d((wt[-2], wt[-1]), stride, 1, group, padding, ia, oa)
        else:
            output = get_layer_node_input_format_conv3d((wt[-3], wt[-2], wt[-1]), stride, 1, group, padding, ia, oa)
        workload[i] = output[0]
        total_macs += output[1]
print(f"GFLOPs: {total_macs*2/1e9}")

with open("mvs_gi.py", "w") as f:
    f.write("workload = {\n")
    for k in workload.keys():
        f.write(str(k)+":")
        f.write(str(workload[k]))
        f.write(",\n")
    f.write("}")