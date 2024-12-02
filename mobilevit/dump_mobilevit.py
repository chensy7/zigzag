from math import ceil

def get_layer_node_input_format_mmm(B, C, K, preds_0, preds_1):
    # convert the data types to precisions based on the onnx definition

    # Equation
    d = {}
    d["operator_type"] = "MatMul"
    d["equation"] = "O[b][k]+=W[k][c]*I[b][c]"

    # Get dimension sizes from input parameters
    K = K
    C = C
    B = B  # Not to be confused with operand 'B' which is the weights
    d["loop_dim_size"] = {"K": K, "C": C, "B": B}
    d["dimension_relations"] = []
    d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
    d["operand_source"] = {"W": [preds_1], "I": [preds_0]}
    d["spatial_mapping"] = None
    d["spatial_mapping_hint"] = None

    return d

def get_layer_node_input_format_conv(
    kernel_shape,
    strides,
    dilations,
    groups,
    padding,
    ia_shape,
    oa_shape,
    preds
):
    d = {}
    d["operator_type"] = "Conv"
    d["equation"] = "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]"
    assert (
        ia_shape[0] == oa_shape[0]
    ), "Batch size is different for input and output activations."
    B = oa_shape[0]
    if B == 0:
        B = 1
    G = groups
    K = ceil(oa_shape[1] / G)
    OX = oa_shape[3]
    OY = oa_shape[2]
    C = ceil(ia_shape[1] / G)
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
        f"ix={strides[0]}*ox+{dilations[0]}*fx",
        f"iy={strides[1]}*oy+{dilations[1]}*fy",
    ]
    d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
    # d["operand_source"] =  {'W': [], 'I': []}
    d["constant_operands"] = ["W"]

    if preds is None: 
        d["operand_source"] = {"I": []}
    else:
        d["operand_source"] = {"I": [preds]}

    d["padding"] = {
        "IY": (padding[0], padding[2]),
        "IX": (padding[1], padding[3]),
    }

    return d

def get_layer_node_input_format_gemm(B, C, K, preds):
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

    d["operand_source"] = {"I": [preds]}

    return d

def convert_into_list(size):
    size = size.split("[")[-1].split("]")[0].split(",")
    return [int(s) for s in size]

workload = {}
with open("mobilevit.txt", "r") as f:
    lines = [line[:-1] for line in f]
    for i in range(len(lines)//3):
        if lines[i*3] == "matmul": 
            c = convert_into_list(lines[i*3+1])[-1]
            b = convert_into_list(lines[i*3+1])[-2]
            k = convert_into_list(lines[i*3+2])[-1]
            workload[i] = get_layer_node_input_format_mmm(b, c, k, i-3, i-2)
        else:  
            ia = convert_into_list(lines[i*3])
            oa = convert_into_list(lines[i*3+2])
            wt = convert_into_list(lines[i*3+1])
            if len(ia) == 4 and ia[0] == 1:
                stride = 2 if oa[-1] < ia[-1] else 1
                group = ia[1] if wt[1] == 1 else 1
                padding = 1 if wt[-1] == 3 else 0
                workload[i] = get_layer_node_input_format_conv((wt[-2], wt[-1]), (stride, stride), (1, 1), group, [padding]*4, ia, oa, i-1 if i>0 else None)
            else:
                b = oa[0]*oa[1]
                k = oa[-1]
                c = wt[-1]
                if lines[i*3][-1] == "q":
                    prev = i - 1
                elif lines[i*3][-1] == "k":
                    prev = i - 2
                elif lines[i*3][-1] == "v":
                    prev = i - 3
                else:
                    prev = i - 1
                workload[i] = get_layer_node_input_format_gemm(b, c, k, prev)

with open("mobilevit.py", "w") as f:
    f.write("workload = {\n")
    for k in workload.keys():
        f.write(str(k)+":")
        f.write(str(workload[k]))
        f.write(",\n")
    f.write("}")