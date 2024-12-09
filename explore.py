from zigzag.api import get_hardware_performance_zigzag, get_hardware_performance_zigzag_without_unused_memory
from zigzag.classes.io.accelerator.parser import AcceleratorParser
from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown
import os
import numpy as np
import pickle
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.inputs.examples.hardware.Base import cores_dut

import argparse

import shutil
import os

from paretoset import paretoset
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="architecture exploration")
parser.add_argument('--parse_output', default=None)
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

## define DoFs
macs = {"D1": [32, 48, 64], "D2": [8, 16, 24], "D3": [8, 16, 24]}
rf_size = {"I": [8], "W": [8], "O": [16]}
rf_dims = {"I": (1, 0, 0), "W": (0, 0, 1), "O": (0, 1, 0)} # spatial mapping fixed, K, C, OX
# rf_dims = {"I": (1, 0, 0), "W": (0, 1, 1), "O": (0, 0, 0)} # spatial mapping fixed, K, OX, OY
mem_size_options = []
for i in range(4, 16, 2):
    mem_size_options += [i*2**18]
l1_size = {"I": mem_size_options, "W": mem_size_options, "O": mem_size_options}

all_exploration_dicts = []
idx = 0
for d1 in macs["D1"]:
    for d2 in macs["D2"]:
        for d3 in macs["D3"]:
            configs_mac = {"D1": d1, "D2": d2, "D3": d3}
            for rf_i in rf_size["I"]:
                for rf_w in rf_size["W"]:
                    for rf_o in rf_size["O"]:
                        for l1_i in l1_size["I"]:
                            for l1_w in l1_size["W"]:
                                for l1_o in l1_size["O"]:
                                    mem_hier = {}
                                    mem_hier["rf_I_size"] = rf_i
                                    mem_hier["rf_W_size"] = rf_w
                                    mem_hier["rf_O_size"] = rf_o
                                    mem_hier["rf_I_bw"] = rf_i
                                    mem_hier["rf_W_bw"] = rf_w
                                    mem_hier["rf_O_bw"] = rf_o
                                    mem_hier["rf_I_dims"] = rf_dims["I"]
                                    mem_hier["rf_W_dims"] = rf_dims["W"]
                                    mem_hier["rf_O_dims"] = rf_dims["O"]
                                    mem_hier["l1_I_size"] = l1_i
                                    mem_hier["l1_W_size"] = l1_w
                                    mem_hier["l1_O_size"] = l1_o
                                    mem_hier["l1_I_bw"] = d2*d3
                                    mem_hier["l1_W_bw"] = d1*d2
                                    mem_hier["l1_O_bw"] = d1*d3
                                    mapping = {
                                        "default": {
                                            "core_allocation": 1,
                                            "spatial_mapping": {'D1': ('K', d1), 'D2': ('C', d2), 'D3': ('OX', d3)},
                                            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
                                        }
                                    }
                                    all_exploration_dicts += [{"idx": idx, "mac": configs_mac, "mem": mem_hier, "mapping": mapping}]
                                    idx += 1

hwarch = 'Base'
# mapping = f"zigzag.inputs.examples.mapping.default"
accelerator = f"zigzag.inputs.examples.hardware.{hwarch}"

def evaluate_single_mem_config(exploration_dict):
    cores = cores_dut(exploration_dict["mac"], exploration_dict["mem"])
    acc_name = os.path.basename(__file__)[:-3]
    accelerator = Accelerator(acc_name, cores)

    os.makedirs(f"outputs/cfg{exploration_dict['idx']}", exist_ok=True)
    dump_filename_pattern=f"outputs/cfg{exploration_dict['idx']}/{hwarch}-{model}-layer_?.json"
    pickle_filename = f"outputs/cfg{exploration_dict['idx']}/{hwarch}-{model}-saved_list_of_cmes.pickle"

    energy, latency, cme = get_hardware_performance_zigzag(workload=workload,
                                                           precision=precision,
                                                           accelerator=accelerator,
                                                           mapping=exploration_dict["mapping"],
                                                           opt=opt,
                                                           dump_filename_pattern=dump_filename_pattern,
                                                           pickle_filename=pickle_filename)

    output_dict = {}
    output_dict["energy"] = energy
    output_dict["latency"] = latency

    # Load in the pickled list of CMEs
    with open(pickle_filename, 'rb') as handle:
        cme_for_all_layers = pickle.load(handle)

    # bar_plot_cost_model_evaluations_breakdown([cme_for_all_layers[1], cme_for_all_layers[-2]], save_path="plot_breakdown.png")  # plot for the first 5 layers

    util = 0
    total = 0
    for cme in cme_for_all_layers:
        total += cme.latency_total2
        util += cme.latency_total2 * cme.MAC_utilization2

    output_dict["energy_by_layer"] = [cme.energy_total for cme in cme_for_all_layers]
    output_dict["latency_by_layer"] = [cme.latency_total2 for cme in cme_for_all_layers]
    output_dict["utilization_by_layer"] = [cme.MAC_utilization2 for cme in cme_for_all_layers]

    pickle_filename = f"summary/{hwarch}-cfg{exploration_dict['idx']}-{model}.pickle"
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

from multiprocessing import Pool

if not args.parse_output:
    with Pool(args.num_cpu) as p:
        p.map(evaluate_single_mem_config, all_exploration_dicts)

def get_mac_mem(cfg):
    macs = cfg["mac"]["D1"] * cfg["mac"]["D2"] * cfg["mac"]["D3"]
    mem = cfg["mem"]["l1_I_size"] + cfg["mem"]["l1_W_size"] + cfg["mem"]["l1_O_size"]
    return macs, mem

def get_area(cfg):
    macs, mem = get_mac_mem(cfg)
    return 367033/1e6*macs/256 + (681300-367033)/1e6, mem/2**22

plt.figure(figsize=(12, 12))
# plt.yscale('log')
area = []
lat = []
energy = []
edp = []
annot = []
util = 0
summary_folder = f"summary"
# if args.parse_output:
#     summary_folder = args.parse_output
for exploration_dict in all_exploration_dicts:
    pickle_filename = f"{summary_folder}/{hwarch}-cfg{exploration_dict['idx']}-{model}.pickle"
    with open(pickle_filename, 'rb') as handle:
        output_dict = pickle.load(handle)
    latency = output_dict['latency']
    e = output_dict['energy']
    logic, mem = get_area(exploration_dict)
    area += [logic+mem]
    lat += [latency]
    energy += [e]
    edp += [latency*e]
    annot += [exploration_dict['idx']]
    plt.annotate(exploration_dict['idx'], (logic+mem, latency*e))
    print(logic+mem, latency)
plt.scatter(area, edp)

# arch = pd.DataFrame({"area": area,
#                     "edp": edp})
# mask = paretoset(arch, sense=["min", "min"])
# pareto = arch[mask]

# pareto_area = pareto.area.tolist()
# pareto_edp = pareto.edp.tolist()
# pareto_index = pareto.index.tolist()

# for idx in pareto_index:
#     exploration_dict = all_exploration_dicts[idx]
#     print(idx, get_mac_mem(exploration_dict), lat[idx], area[idx], energy[idx])

# plt.scatter(pareto_area, pareto_edp)
plt.savefig(f"{model}-summary.pdf")
plt.show()