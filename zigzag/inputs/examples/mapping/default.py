mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 64), "D2": ("OX", 32), "D3": ("OY", 4)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "Add": {
        "core_allocation": 1,
        # "spatial_mapping": {},
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
    },
    "Pooling": {
        "core_allocation": 1,
        # "spatial_mapping": {},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
}
