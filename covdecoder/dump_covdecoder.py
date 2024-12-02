def get_layer_node_input_format_gemm(B, C, K):
    # convert the data types to precisions based on the onnx definition

    # Equation
    d = {}
    d["operator_type"] = "GEMM"
    d["equation"] = "O[b][k]+=W[k][c]*I[b][c]"

    # Get dimension sizes from input parameters
    K = K
    C = C
    B = B  # Not to be confused with operand 'B' which is the weights
    d["loop_dim_size"] = {"K": K, "C": C, "B": B}
    d["dimension_relations"] = []
    d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
    d["operand_source"] = {"W": [], "I": []}
    d["constant_operands"] = ["W"]

    d["operand_source"] = {"I": []}

    return d

def get_layer_node_input_format_conv(
    B,K,G,IX,IY,OX,OY,C,FX,FY,strides,dilations,padding
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

    d["operand_source"] = {"I": []}

    d["padding"] = {
        "IY": (padding, padding),
        "IX": (padding, padding),
    }

    return d

def convert_into_list(size):
    size = size.split("[")[-1].split("]")[0].split(",")
    return [int(s) for s in size]

workload = {}
with open("covdecoder.txt", "r") as f:
    lines = [line[:-1] for line in f]
    for i in range(len(lines)):
        line_list = convert_into_list(lines[i])
        if len(line_list) > 3:
            workload[i] = get_layer_node_input_format_conv(*line_list)
        else:
            workload[i] = get_layer_node_input_format_gemm(*line_list)

with open("covdecoder.py", "w") as f:
    f.write("workload = {\n")
    for k in workload.keys():
        f.write(str(k)+":")
        f.write(str(workload[k]))
        f.write(",\n")
    f.write("}")