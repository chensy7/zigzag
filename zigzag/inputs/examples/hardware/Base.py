# author: schen

import os
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core

def memory_hierarchy_dut(multiplier_array, mem_hier, visualize=False):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """

    if mem_hier["rf_W_size"] != 0:
        reg_W = MemoryInstance(
            name="rf_W",
            size=mem_hier["rf_W_size"],
            r_bw=mem_hier["rf_W_bw"],
            w_bw=mem_hier["rf_W_bw"],
            r_cost=0,
            w_cost=0,
            area=0,
            r_port=1,
            w_port=1,
            rw_port=0,
            latency=1,
        )

    if mem_hier["rf_I_size"] != 0:
        reg_I = MemoryInstance(
            name="rf_I",
            size=mem_hier["rf_I_size"],
            r_bw=mem_hier["rf_I_bw"],
            w_bw=mem_hier["rf_I_bw"],
            r_cost=0,
            w_cost=0,
            area=0,
            r_port=1,
            w_port=1,
            rw_port=0,
            latency=1,
        )

    if mem_hier["rf_O_size"] != 0:
        reg_O = MemoryInstance(
            name="rf_O",
            size=mem_hier["rf_O_size"],
            r_bw=mem_hier["rf_O_bw"],
            w_bw=mem_hier["rf_O_bw"],
            r_cost=0.00087790554*24,
            w_cost=0.00087790554*24,
            area=0,
            r_port=1,
            w_port=1,
            rw_port=0,
            latency=1,
        )

    ##################################### on-chip memory hierarchy building blocks #####################################
    SRAM_ENERGY_32b = 5.9*2
    if mem_hier["l1_W_size"] != 0:
        L1_W = MemoryInstance(
            name="L1_W",
            size=mem_hier["l1_W_size"],
            r_bw=mem_hier["l1_W_bw"],
            w_bw=mem_hier["l1_W_bw"],
            r_cost=SRAM_ENERGY_32b * mem_hier["l1_W_bw"] / 32, # tsmc28 memory compiler for 2048x32
            w_cost=SRAM_ENERGY_32b * mem_hier["l1_W_bw"] / 32,
            area=0,
            r_port=0,
            w_port=0,
            rw_port=1,
            latency=1,
            min_r_granularity=8,
            min_w_granularity=8,
        )

    if mem_hier["l1_I_size"] != 0:
        L1_I = MemoryInstance(
            name="L1_I",
            size=mem_hier["l1_I_size"],
            r_bw=mem_hier["l1_I_bw"],
            w_bw=mem_hier["l1_I_bw"],
            r_cost=SRAM_ENERGY_32b * mem_hier["l1_I_bw"] / 32, # tsmc28 memory compiler for 2048x32
            w_cost=SRAM_ENERGY_32b * mem_hier["l1_I_bw"] / 32,
            area=0,
            r_port=0,
            w_port=0,
            rw_port=1,
            latency=1,
            min_r_granularity=8,
            min_w_granularity=8,
        )

    if mem_hier["l1_O_size"] != 0:
        L1_O = MemoryInstance(
            name="L1_O",
            size=mem_hier["l1_O_size"],
            r_bw=mem_hier["l1_O_bw"],
            w_bw=mem_hier["l1_O_bw"],
            r_cost=SRAM_ENERGY_32b * mem_hier["l1_O_bw"] / 32, # tsmc28 memory compiler for 2048x32
            w_cost=SRAM_ENERGY_32b * mem_hier["l1_O_bw"] / 32,
            area=0,
            r_port=1,
            w_port=1,
            rw_port=0,
            latency=1,
            min_r_granularity=8,
            min_w_granularity=8,
        )

    #######################################################################################################################

    dram = MemoryInstance(
        name="dram",
        size=10000000000,
        r_bw=128,
        w_bw=128,
        r_cost=128*20, #assume 10 pJ/bit
        w_cost=128*20,
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
    if mem_hier["rf_I_size"] != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=reg_I,
            operands=("I1",),
            port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
            served_dimensions={mem_hier["rf_I_dims"]},
        )
    if mem_hier["rf_W_size"] != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=reg_W,
            operands=("I2",),
            port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
            served_dimensions={mem_hier["rf_W_dims"]},
        )
    if mem_hier["rf_O_size"] != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=reg_O,
            operands=("O",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
            ),
            served_dimensions={mem_hier["rf_O_dims"]},
        )

    ##################################### on-chip highest memory hierarchy initialization #####################################
    if mem_hier["l1_I_size"] != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=L1_I,
            operands=("I1",),
            port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},),
            served_dimensions="all",
        )

    if mem_hier["l1_W_size"] != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=L1_W,
            operands=("I2",),
            port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},),
            served_dimensions="all",
        )

    if mem_hier["l1_O_size"] != 0:
        memory_hierarchy_graph.add_memory(
            memory_instance=L1_O,
            operands=("O",),
            port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},),
            served_dimensions="all",
        )

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


def multiplier_array_dut(mul_dims):
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.477 # scaled from isscc14 horowitz
    multiplier_area = 1

    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, mul_dims)

    return multiplier_array


def cores_dut(mul_dims, memory_hier):
    multiplier_array1 = multiplier_array_dut(mul_dims)
    memory_hierarchy1 = memory_hierarchy_dut(multiplier_array1, memory_hier)

    core1 = Core(1, multiplier_array1, memory_hierarchy1)

    return {core1}
