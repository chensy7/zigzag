import time
import multiprocessing
import random

import numpy as np

from deap import algorithms, base, creator, tools

from zigzag.api import get_hardware_performance_zigzag, get_hardware_performance_zigzag_without_unused_memory
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.inputs.examples.hardware.Base import cores_dut

import argparse

import shutil
import os

import logging
logging.disable(logging.CRITICAL)

parser = argparse.ArgumentParser(description="architecture exploration")
parser.add_argument('--model')
args = parser.parse_args()

shutil.rmtree('summary', ignore_errors=True); os.mkdir("summary")
shutil.rmtree('outputs', ignore_errors=True); os.mkdir("outputs")

spatial_map_choices = [
    ('K', 4), ('K', 8), ('K', 12), ('K', 16), #('K', 20), ('K', 24), ('K', 28), ('K', 32), ('K', 36), ('K', 40),
    ('C', 4), ('C', 8), ('C', 12), ('C', 16), #('C', 20), ('C', 24), ('C', 28), ('C', 32), ('C', 36), ('C', 40),
    ('OX', 4), ('OX', 8), ('OX', 12), ('OX', 16), #('OX', 20), ('OX', 24), ('OX', 28), ('OX', 32), ('OX', 36), ('OX', 40),
    ('FX', 3),
]   
mem_size_choices = []
for i in range(0, 16):
    mem_size_choices += [i*2**18]
int_max_list = [len(spatial_map_choices)-1]*4 + [len(mem_size_choices)-1]*3

opt = 'latency'
model = args.model
if ".py" in model:
    model = model.split(".")[0]
    workload = f"zigzag.inputs.examples.workload.{model}"
else:
    onnx_model_path = f"zigzag/inputs/examples/workload/{model}.onnx"
    workload = onnx_model_path
precision = f"zigzag/inputs/examples/workload/{model}.json"
hwarch = 'Base'
accelerator = f"zigzag.inputs.examples.hardware.{hwarch}"

def evaluate_single_config(individual):
    exploration_dict = {}
    D1, d1 = spatial_map_choices[individual[0]]
    D2, d2 = spatial_map_choices[individual[1]]
    D3, d3 = spatial_map_choices[individual[2]]
    D4, d4 = spatial_map_choices[individual[3]]
    configs_mac = {"D1": d1, "D2": d2, "D3": d3, "D4": d4}
    # print(configs_mac)
    mapping = {
        "default": {
            "core_allocation": 1,
            "spatial_mapping": {'D1': (D1, d1), 'D2': (D2, d2), 'D3': (D3, d3), 'D4': (D4, d4)},
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        }
    }
    # print(mapping)
    mem_hier = {}
    mem_hier["rf_I_size"] = 8
    mem_hier["rf_W_size"] = 8
    mem_hier["rf_O_size"] = 16
    mem_hier["rf_I_bw"] = 8
    mem_hier["rf_W_bw"] = 8
    mem_hier["rf_O_bw"] = 16
    D_I_relevance = [0 if D in ['C', 'OX', 'OY', 'FX'] else 1 for D in [D1, D2, D3, D4]]
    D_W_relevance = [0 if D in ['C', 'K', 'FX'] else 1 for D in [D1, D2, D3, D4]]
    D_O_relevance = [0 if D in ['K', 'OX', 'OY'] else 1 for D in [D1, D2, D3, D4]]
    mem_hier["rf_I_dims"] = tuple(D_I_relevance)
    mem_hier["rf_W_dims"] = tuple(D_W_relevance)
    mem_hier["rf_O_dims"] = tuple(D_O_relevance)
    mem_hier["l1_I_size"] = mem_size_choices[individual[4]]
    mem_hier["l1_W_size"] = mem_size_choices[individual[5]]
    mem_hier["l1_O_size"] = mem_size_choices[individual[6]]
    spatial_sizes = [d1, d2, d3, d4]
    def get_max_bw_required(relevance, sizes):
        bw = 1
        for i in range(len(relevance)):
            if relevance[i] == 0:
                bw *= sizes[i]
        return bw
    mem_hier["l1_I_bw"] = get_max_bw_required(D_I_relevance, spatial_sizes)
    mem_hier["l1_W_bw"] = get_max_bw_required(D_W_relevance, spatial_sizes)
    mem_hier["l1_O_bw"] = get_max_bw_required(D_O_relevance, spatial_sizes)

    cores = cores_dut(configs_mac, mem_hier)
    acc_name = os.path.basename(__file__)[:-3]
    accelerator = Accelerator(acc_name, cores)

    idx = hash(tuple(individual))
    os.makedirs(f"outputs/cfg{idx}", exist_ok=True)
    dump_filename_pattern=f"outputs/cfg{idx}/{hwarch}-{model}-layer_?.json"
    pickle_filename = f"outputs/cfg{idx}/{hwarch}-{model}-saved_list_of_cmes.pickle"

    try:
        energy, latency, cme = get_hardware_performance_zigzag(workload=workload,
                                                               precision=precision,
                                                               accelerator=accelerator,
                                                               mapping=mapping,
                                                               opt=opt,
                                                               dump_filename_pattern=dump_filename_pattern,
                                                               pickle_filename=pickle_filename)
    except:
        latency = 1e32
    # print(latency)
    return (latency,)

def mutate_single_config(individual):
    # randomly change one config
    index = random.randint(0, len(individual)-1)
    individual[index] = random.randint(0, int_max_list[index])
    return (individual,)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def initIndividual(icls):
    config = []
    for i in range(len(int_max_list)):
        config += [random.randint(0, int_max_list[i])]
    return icls(config)

toolbox = base.Toolbox()

# for i, int_max in enumerate(int_max_list):
#     toolbox.register(f"attr_int{i}", random.randint, 0, int_max)
# toolbox.register("individual", tools.initCycle, creator.Individual,
#                  (toolbox.attr_int0, toolbox.attr_int1, toolbox.attr_int2, toolbox.attr_int3, toolbox.attr_int4, toolbox.attr_int5, toolbox.attr_int6, toolbox.attr_int7, toolbox.attr_int8, toolbox.attr_int9, toolbox.attr_int10), n=1)

toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", evaluate_single_config)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_single_config)
toolbox.register("select", tools.selTournament, tournsize=3)

random.seed(64)
cpu_count = multiprocessing.cpu_count()
print(f"CPU count: {cpu_count}")
pool = multiprocessing.Pool(cpu_count)
toolbox.register("map", pool.map)

POPULATION_SIZE = 32
NUM_GEN = 400
pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=NUM_GEN, stats=stats, halloffame=hof)

pool.close()