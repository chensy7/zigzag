from zigzag.api import get_hardware_performance_zigzag, get_hardware_performance_zigzag_without_unused_memory
from zigzag.classes.io.accelerator.parser import AcceleratorParser
from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown
import os
import numpy as np
import pickle
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.inputs.examples.hardware.Template_2L import cores_dut

import argparse

import shutil
import os

def factorize(num):
    if num == 0:
        return [0]
    else:
        return [n for n in range(1, num + 1) if num % n == 0]

parser = argparse.ArgumentParser(description="architecture exploration")
parser.add_argument('--parse_output', action="store_true")
parser.add_argument('--model')
parser.add_argument('--num_cpu', default=1, type=int)
args = parser.parse_args()

if not args.parse_output:
    shutil.rmtree('summary', ignore_errors=True); os.mkdir("summary")
    shutil.rmtree('outputs', ignore_errors=True); os.mkdir("outputs")

opt = 'EDP'
model = args.model
if ".py" in model:
    model = model.split(".")[0]
    workload = f"zigzag.inputs.examples.workload.{model}"
else:
    onnx_model_path = f"zigzag/inputs/examples/workload/{model}.onnx"
    workload = onnx_model_path
precision = f"zigzag/inputs/examples/workload/{model}.json"

energies_all = {}
latecies_all = {}
energy_dict = {}
latency_dict = {}

# architecture exploration
sram_mats = {} # key (width, depth), value (number of insts, energy)
sram_mats[(32, 256)] = (8, )
sram_mats[(64, 4096)] = (8, )
sram_mats[(256, 64)] = (8, )

# getting search space for SRAM
total_sram_budget = 8* 1024**2  # 2MB
l1_sram_base_size = 8192 * 8 *16 # 32KB
max_l1_sram_size_multiplier = total_sram_budget // l1_sram_base_size
l1_size_multiplier_choices = np.append(np.flip(2**np.arange(np.log(max_l1_sram_size_multiplier)//np.log(2)+1)), 0)
l1_size_multipliers = []
for idx_l1_w_s, l1_w_s in enumerate(l1_size_multiplier_choices):
    l1_sram_w_size = l1_sram_base_size * l1_w_s
    for idx_l1_i_s, l1_i_s in enumerate(np.flip(l1_size_multiplier_choices)):
        l1_sram_i_size = l1_sram_base_size * l1_i_s
        if l1_sram_w_size + l1_sram_i_size < total_sram_budget:
            l1_o_s = (total_sram_budget - l1_sram_w_size - l1_sram_i_size) // l1_sram_base_size
        else:
            l1_o_s = 0
            l1_i_s = (total_sram_budget - l1_sram_w_size) // l1_sram_base_size
            l1_size_multipliers += [(int(l1_w_s), int(l1_i_s), int(l1_o_s))]
            break
        l1_size_multipliers += [(int(l1_w_s), int(l1_i_s), int(l1_o_s))]

l1_multipliers = []
for (l1_w_s, l1_i_s, l1_o_s) in l1_size_multipliers:
    for l1_w_bw in factorize(l1_w_s):
        for l1_i_bw in factorize(l1_i_s):
            for l1_o_bw in factorize(l1_o_s):
                l1_multipliers += [(l1_w_s, l1_i_s, l1_o_s, l1_w_bw, l1_i_bw, l1_o_bw)]

all_exploration_dicts = []
idx = 0
# l1_multipliers = [l1_multipliers[35]]
reg_multipliers = [(1, (0, 0), 1, (0, 0), 1, (0, 0), 1, 1, 1)] 
for reg in reg_multipliers:
    for l1 in l1_multipliers:
        memory_hierarchy_exploration_dict = {}
        memory_hierarchy_exploration_dict['L1_SRAM_BASE_SIZE'] = l1_sram_base_size
        memory_hierarchy_exploration_dict['idx'] = idx
        memory_hierarchy_exploration_dict['REG_W_SIZE_MULTIPLIER'] = reg[4]
        memory_hierarchy_exploration_dict['REG_I_SIZE_MULTIPLIER'] = reg[2]
        memory_hierarchy_exploration_dict['REG_O_SIZE_MULTIPLIER'] = reg[0]
        memory_hierarchy_exploration_dict['REG_W_BW_MULTIPLIER'] = reg[-1]
        memory_hierarchy_exploration_dict['REG_I_BW_MULTIPLIER'] = reg[-2]
        memory_hierarchy_exploration_dict['REG_O_BW_MULTIPLIER'] = reg[-3]
        memory_hierarchy_exploration_dict['REG_W_DIM'] = reg[5]
        memory_hierarchy_exploration_dict['REG_I_DIM'] = reg[3]
        memory_hierarchy_exploration_dict['REG_O_DIM'] = reg[1]
        memory_hierarchy_exploration_dict['L1_W_SIZE_MULTIPLIER'] = l1[0]
        memory_hierarchy_exploration_dict['L1_I_SIZE_MULTIPLIER'] = l1[1]
        memory_hierarchy_exploration_dict['L1_O_SIZE_MULTIPLIER'] = l1[2]
        memory_hierarchy_exploration_dict['L1_W_BW_MULTIPLIER'] = l1[3]
        memory_hierarchy_exploration_dict['L1_I_BW_MULTIPLIER'] = l1[4]
        memory_hierarchy_exploration_dict['L1_O_BW_MULTIPLIER'] = l1[5]
        idx += 1
        all_exploration_dicts += [memory_hierarchy_exploration_dict]
        
hwarch = 'Template_2L'
mapping = f"zigzag.inputs.examples.mapping.default"
accelerator = f"zigzag.inputs.examples.hardware.{hwarch}"

def evaluate_single_mem_config(mem_dict):
    cores = cores_dut(mem_dict)
    acc_name = os.path.basename(__file__)[:-3]
    accelerator = Accelerator(acc_name, cores)

    dump_filename_pattern=f"outputs/{hwarch}-mem{mem_dict['idx']}-{model}-layer_?.json"
    pickle_filename = f"outputs/{hwarch}-mem{mem_dict['idx']}-{model}-saved_list_of_cmes.pickle"

    energy, latency, cme = get_hardware_performance_zigzag_without_unused_memory(workload=workload,
                                                           precision=precision,
                                                           accelerator=accelerator,
                                                           mapping=mapping,
                                                           opt=opt,
                                                           dump_filename_pattern=dump_filename_pattern,
                                                           pickle_filename=pickle_filename)

    output_dict = {}
    output_dict["energy"] = energy
    output_dict["latency"] = latency
    # print(energy*latency)

    # Load in the pickled list of CMEs
    with open(pickle_filename, 'rb') as handle:
        cme_for_all_layers = pickle.load(handle)

    bar_plot_cost_model_evaluations_breakdown([cme_for_all_layers[1], cme_for_all_layers[-2]], save_path="plot_breakdown.png")  # plot for the first 5 layers

    util = 0
    total = 0
    for cme in cme_for_all_layers:
        total += cme.latency_total2
        util += cme.latency_total2 * cme.MAC_utilization2
        print(cme.layer, cme.spatial_mapping['O'])
    #     print(cme.MAC_utilization2, cme.energy_total, cme.latency_total2)
    # print(util/total)

    output_dict["energy_by_layer"] = [cme.energy_total for cme in cme_for_all_layers]
    output_dict["latency_by_layer"] = [cme.latency_total2 for cme in cme_for_all_layers]
    output_dict["utilization_by_layer"] = [cme.MAC_utilization2 for cme in cme_for_all_layers]

    pickle_filename = f"summary/{hwarch}-mem{mem_dict['idx']}-{model}.pickle"
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

from multiprocessing import Pool

if not args.parse_output:
    with Pool(args.num_cpu) as p:
        p.map(evaluate_single_mem_config, all_exploration_dicts)

min_edp = 1e25
min_edp_idx = -1
max_edp = 0
min_energy = 1e25
min_latency = 1e25
edp_by_layer = []
energy_by_layer = []
latency_by_layer = []
util_by_layer = []
for mem_dict in all_exploration_dicts:
    pickle_filename = f"summary/{hwarch}-mem{mem_dict['idx']}-{model}.pickle"
    with open(pickle_filename, 'rb') as handle:
        output_dict = pickle.load(handle)
    energy = output_dict['energy']
    latency = output_dict['latency']
    edp = energy * latency
    energy_by_layer += [[output_dict["energy_by_layer"][i] for i in range(len(output_dict["energy_by_layer"]))]]
    latency_by_layer += [[output_dict["latency_by_layer"][i] for i in range(len(output_dict["energy_by_layer"]))]]
    util_by_layer += [[output_dict["utilization_by_layer"][i] for i in range(len(output_dict["utilization_by_layer"]))]]
    edp_by_layer += [[output_dict["energy_by_layer"][i]*output_dict["latency_by_layer"][i] for i in range(len(output_dict["energy_by_layer"]))]]
    # edp = np.sum(np.array([output_dict["energy_by_layer"][i]*output_dict["latency_by_layer"][i] for i in range(len(output_dict["energy_by_layer"]))]))
    if edp < min_edp:
        min_edp = edp
        min_edp_idx = mem_dict["idx"]
    if energy < min_energy:
        min_energy = energy
    if latency < min_latency:
        min_latency = latency
    if edp > max_edp:
        max_edp = edp
best_energy = 0
best_latency = 0
best_edp_sum = 0
best_util = 0
for i in range(len(edp_by_layer[0])):
    edp_all_arch = [edp_by_layer[j][i] for j in range(len(edp_by_layer))]
    best_edp_all_arch = min(edp_all_arch)
    best_edp_index = np.argmin(np.array(edp_all_arch))
    print(f"Layer {i}, config {best_edp_index},  min edp {best_edp_all_arch:.2e}, edp of idx {min_edp_idx} is {edp_all_arch[min_edp_idx]:.2e}, max edp {max(edp_all_arch):.2e}")
    best_energy += energy_by_layer[best_edp_index][i]
    best_latency += latency_by_layer[best_edp_index][i]
    best_util += util_by_layer[best_edp_index][i] * latency_by_layer[best_edp_index][i]
    best_edp_sum += best_edp_all_arch
best_edp = best_energy * best_latency
print(best_util/best_latency)
print(f"Best edp = {best_edp:.2e} pJ*cycles Best energy = {best_energy:.2e} pJ Best latency = {best_latency:.2e} cycles")
print(f"Min energy = {min_energy:.2e} pJ", f"Min latency = {min_latency:.2e} cycles")
print(f"Max edp = {max_edp:.2e} pJ*cycles", f"Min edp = {min_edp:.2e} pJ*cycles", f"Min edp idx {min_edp_idx} config {all_exploration_dicts[min_edp_idx]}")