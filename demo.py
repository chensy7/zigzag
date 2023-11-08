from zigzag.api import get_hardware_performance_zigzag
from zigzag.classes.io.accelerator.parser import AcceleratorParser
import os

opt = 'EDP'
model = "alexnet"
onnx_model_path = f"zigzag/inputs/examples/workload/{model}.onnx"
workload = onnx_model_path
precision = f"zigzag/inputs/examples/workload/{model}.json"

energies_all = {}
latecies_all = {}
energy_dict = {}
latency_dict = {}

# for hwarch in ["Edge_TPU", "Tesla_NPU", "Meta_prototype", "TPU", "Ascend"]:
for hwarch in ["Edge_TPU_like"]:
    mapping = f"zigzag.inputs.examples.mapping.default"
    accelerator = f"zigzag.inputs.examples.hardware.{hwarch}"

    dump_filename_pattern=f"outputs/{hwarch}-{model}-layer_?.json"
    pickle_filename = f"outputs/{hwarch}-{model}-saved_list_of_cmes.pickle"

    energy, latency, cme = get_hardware_performance_zigzag(workload=workload,
                                                           precision=precision,
                                                           accelerator=accelerator,
                                                           mapping=mapping,
                                                           opt=opt,
                                                           dump_filename_pattern=dump_filename_pattern,
                                                           pickle_filename=pickle_filename)
    print(f"Total network energy = {energy:.2e} pJ")
    print(f"Total network latency = {latency:.2e} cycles")
    print(f"Total edp = {energy*latency:.2e} pJ*cycles")

    energy_dict[hwarch] = energy
    latency_dict[hwarch] = latency

    import pickle
    from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown

    # Load in the pickled list of CMEs
    with open(pickle_filename, 'rb') as fp:
        cme_for_all_layers = pickle.load(fp)

    energies_all[hwarch] = [cme.energy_total for cme in cme_for_all_layers]
    latecies_all[hwarch] = [cme.latency_total2 for cme in cme_for_all_layers]

    # Plot all the layers and save to 'plot_all.png'
    # bar_plot_cost_model_evaluations_breakdown(cme_for_all_layers[:5], save_path="plot_breakdown.png")  # plot for the first 5 layers
    # bar_plot_cost_model_evaluations_breakdown(cme_for_all_layers, save_path=f"{hwarch}-{model}-plot_breakdown.png")  # uncomment this line to plot for all the layers

import numpy as np

length = len(energies_all["Edge_TPU"])
latency_total = 0
energy_total = 0
for i in range(length):
    print(i)
    latency_cur = []
    energy_cur = []
    edp_cur = []
    for k in latecies_all.keys():
        latency_cur += [latecies_all[k][i]]
        energy_cur += [energies_all[k][i]]
        edp_cur += [latecies_all[k][i]*energies_all[k][i]]
    if opt == 'energy':
        idx = np.argmin(energy_cur)
    elif opt == 'latency':
        idx = np.argmin(latency_cur)
    else:
        idx = np.argmin(edp_cur)
    print(list(latecies_all.keys())[idx])
    latency_total += latency_cur[idx]
    energy_total += energy_cur[idx]
for k in energy_dict.keys():
    print(k)
    print(f"Total network energy = {energy_dict[k]:.2e} pJ")
    print(f"Total network latency = {latency_dict[k]:.2e} cycles")
    print(f"Total edp = {energy_dict[k]*latency_dict[k]:.2e} pJ*cycles")
print("Best")
print(f"Total network energy = {energy_total:.2e} pJ")
print(f"Total network latency = {latency_total:.2e} cycles")
print(f"Total edp = {energy_total*latency_total:.2e} pJ*cycles")


# from zigzag.visualization.results.print_mapping import print_mapping
# from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph

# visualize_memory_hierarchy_graph(cme_for_all_layers[0].accelerator.cores[0].memory_hierarchy)
# for cme in cme_for_all_layers:
#     print_mapping(cme)