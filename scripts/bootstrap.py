import hashlib

import numpy as np

# Deterministic bootstrap scheme shared by every component that needs to know
# which rows a tree was trained on (worker training, evaluation): the same
# (model_id, tree_index) always yields the same sample, with no need to
# persist or transmit it.


def tree_seed(model_id: str, tree_index: int) -> int:
    """32-bit seed derived from (model_id, tree_index). Python's hash() is
    randomized per process, so a real hash function is used."""
    digest = hashlib.sha256(f"{model_id}:{tree_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def generate_bootstrap_indices(model_id: str, tree_index: int, n_rows: int) -> np.ndarray:
    """Row positions (with replacement, n_rows of them) forming the bootstrap
    sample of one tree. Positions refer to the training set as stored."""
    rng = np.random.default_rng(tree_seed(model_id, tree_index))
    return rng.integers(0, n_rows, size=n_rows)
