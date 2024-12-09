mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 16), "D2": ("C", 16), "D3": ("FX", 3), "D4": ("OX", 40)},
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
