# author: schen

import os
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core

def memory_hierarchy_dut(multiplier_array, memory_hierarchy_exploration_dict, visualize=False):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """

    REG_W_SIZE_MULTIPLIER = memory_hierarchy_exploration_dict['REG_W_SIZE_MULTIPLIER']
    REG_I_SIZE_MULTIPLIER = memory_hierarchy_exploration_dict['REG_I_SIZE_MULTIPLIER']
    REG_O_SIZE_MULTIPLIER = memory_hierarchy_exploration_dict['REG_O_SIZE_MULTIPLIER']
    REG_W_BW_MULTIPLIER = memory_hierarchy_exploration_dict['REG_W_BW_MULTIPLIER']
    REG_I_BW_MULTIPLIER = memory_hierarchy_exploration_dict['REG_I_BW_MULTIPLIER']
    REG_O_BW_MULTIPLIER = memory_hierarchy_exploration_dict['REG_O_BW_MULTIPLIER']
    REG_W_DIM = memory_hierarchy_exploration_dict['REG_W_DIM']
    REG_I_DIM = memory_hierarchy_exploration_dict['REG_I_DIM']
    REG_O_DIM = memory_hierarchy_exploration_dict['REG_O_DIM']
    L1_W_SIZE_MULTIPLIER = memory_hierarchy_exploration_dict['L1_W_SIZE_MULTIPLIER']
    L1_I_SIZE_MULTIPLIER = memory_hierarchy_exploration_dict['L1_I_SIZE_MULTIPLIER']
    L1_O_SIZE_MULTIPLIER = memory_hierarchy_exploration_dict['L1_O_SIZE_MULTIPLIER']
    L1_W_BW_MULTIPLIER = memory_hierarchy_exploration_dict['L1_W_BW_MULTIPLIER']
    L1_I_BW_MULTIPLIER = memory_hierarchy_exploration_dict['L1_I_BW_MULTIPLIER']
    L1_O_BW_MULTIPLIER = memory_hierarchy_exploration_dict['L1_O_BW_MULTIPLIER']
    L1_SRAM_BASE_SIZE = memory_hierarchy_exploration_dict['L1_SRAM_BASE_SIZE']

    if REG_W_SIZE_MULTIPLIER != 0:
        reg_W = MemoryInstance(
            name="rf_W",
            size=8 * REG_W_SIZE_MULTIPLIER,
            r_bw=8 * REG_W_BW_MULTIPLIER,
            w_bw=8 * REG_W_BW_MULTIPLIER,
            r_cost=0 * REG_W_BW_MULTIPLIER,
            w_cost=0 * REG_W_BW_MULTIPLIER,
            area=0,
            r_port=1,
            w_port=1,
            rw_port=0,
            latency=1,
        )

    if REG_I_SIZE_MULTIPLIER != 0:
        reg_I = MemoryInstance(
            name="rf_I",
            size=8 * REG_I_SIZE_MULTIPLIER,
            r_bw=8 * REG_I_BW_MULTIPLIER,
            w_bw=8 * REG_I_BW_MULTIPLIER,
            r_cost=0 * REG_I_BW_MULTIPLIER,
            w_cost=0 * REG_I_BW_MULTIPLIER,
            area=0,
            r_port=1,
            w_port=1,
            rw_port=0,
            latency=1,
        )

    if REG_O_SIZE_MULTIPLIER != 0:
        reg_O = MemoryInstance(
            name="rf_O",
            size=16,
            r_bw=16,
            w_bw=16,
            r_cost=0.00087790554*24,
            w_cost=0.00087790554*24,
            area=0,
            r_port=2,
            w_port=2,
            rw_port=0,
            latency=1,
        )

    ##################################### on-chip memory hierarchy building blocks #####################################
    SRAM_ENERGY_32b = 5.9
    if L1_W_SIZE_MULTIPLIER != 0:
        L1_W = MemoryInstance(
            name="L1_W",
            size=4096*512*4,
            r_bw=6144,
            w_bw=6144,
            r_cost=26.5631e-6*500*4*0.9*2e-9*1e12,
            w_cost=30.1536e-6*500*4*0.9*2e-9*1e12,
            area=0,
            r_port=0,
            w_port=0,
            rw_port=1,
            latency=1,
            min_r_granularity=8,
            min_w_granularity=8,
        )

    if L1_I_SIZE_MULTIPLIER != 0:
        L1_I = MemoryInstance(
            name="L1_I",
            size=8192 * 136*4,
            r_bw=5504,
            w_bw=5504,
            r_cost=38.5400e-6*500*1*0.9*2e-9*1e12,
            w_cost=46.0806e-6*500*1*0.9*2e-9*1e12,
            area=0,
            r_port=0,
            w_port=0,
            rw_port=1,
            latency=1,
            min_r_granularity=8,
            min_w_granularity=8,
        )

    if L1_O_SIZE_MULTIPLIER != 0:
        L1_O = MemoryInstance(
            name="L1_O",
            size=8192*336*4,
            r_bw=10240,
            w_bw=10240,
            r_cost=32.1994e-6*500*3*0.9*2e-9*1e12, # tsmc28 memory compiler for 2048x32
            w_cost=38.3185e-6*500*3*0.9*2e-9*1e12,
            area=0,
            r_port=0,
            w_port=0,
            rw_port=1,
            latency=1,
            min_r_granularity=8,
            min_w_granularity=8,
        )

    # sram_2M_with_16_128K_bank_128_1r_1w = MemoryInstance(
    #     name="sram_2MB",
    #     size=131072 * 16 * 8,
    #     r_bw=128 * 16,
    #     w_bw=128 * 16,
    #     r_cost=26.01 * 16,
    #     w_cost=23.65 * 16,
    #     area=0,
    #     r_port=1,
    #     w_port=1,
    #     rw_port=0,
    #     latency=1,
    #     min_r_granularity=64,
    #     min_w_granularity=64,
    # )

    #######################################################################################################################

    dram = MemoryInstance(
        name="dram",
        size=10000000000,
        r_bw=64*2,
        w_bw=64*2,
        r_cost=640*2, #assume 10 pJ/bit
        w_cost=640*2,
        area=0,
        r_port=0,
        w_port=0,
        rw_port=1,
        latency=1,
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    if REG_I_SIZE_MULTIPLIER != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=reg_I,
            operands=("I1",),
            port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
            served_dimensions={REG_I_DIM},
        )
    if REG_W_SIZE_MULTIPLIER != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=reg_W,
            operands=("I2",),
            port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
            served_dimensions={REG_W_DIM},
        )
    if REG_O_SIZE_MULTIPLIER != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=reg_O,
            operands=("O",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
            ),
            served_dimensions={REG_O_DIM},
        )

    # memory_hierarchy_graph.add_memory(
    #     memory_instance=reg_W_128B,
    #     operands=("I2",),
    #     port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
    #     served_dimensions={(0, 0)},
    # )
    # memory_hierarchy_graph.add_memory(
    #     memory_instance=reg_O_2B,
    #     operands=("O",),
    #     port_alloc=(
    #         {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
    #     ),
    #     served_dimensions={(0, 1)},
    # )

    ##################################### on-chip highest memory hierarchy initialization #####################################
    if L1_I_SIZE_MULTIPLIER != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=L1_I,
            operands=("I1",),
            port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},),
            served_dimensions="all",
        )

    if L1_W_SIZE_MULTIPLIER != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=L1_W,
            operands=("I2",),
            port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},),
            served_dimensions="all",
        )

    if L1_O_SIZE_MULTIPLIER != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=L1_O,
            operands=("O",),
            port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": "rw_port_1", "th": "rw_port_1"},),
            served_dimensions="all",
        )

    # memory_hierarchy_graph.add_memory(
    #     memory_instance=sram_2M_with_16_128K_bank_128_1r_1w,
    #     operands=("I1", "O"),
    #     port_alloc=(
    #         {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
    #         {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
    #     ),
    #     served_dimensions="all",
    # )

    ####################################################################################################################

    memory_hierarchy_graph.add_memory(
        memory_instance=dram,
        operands=("I1", "I2", "O"),
        port_alloc=(
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
            {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
            {
                "fh": "rw_port_1",
                "tl": "rw_port_1",
                "fl": "rw_port_1",
                "th": "rw_port_1",
            },
        ),
        served_dimensions="all",
    )
    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def multiplier_array_dut():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.477 # 174e-3*(158737*2e-9)/115605504*1e12
    multiplier_area = 1
    dimensions = {"D1": 16, "D2": 16, "D3": 3, "D4": 40}  # {'D1': ('K', 32), 'D2': ('C', 32)}

    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def cores_dut(memory_hierarchy_exploration_dict):
    multiplier_array1 = multiplier_array_dut()
    memory_hierarchy1 = memory_hierarchy_dut(multiplier_array1, memory_hierarchy_exploration_dict)

    core1 = Core(1, multiplier_array1, memory_hierarchy1)

    return {core1}
