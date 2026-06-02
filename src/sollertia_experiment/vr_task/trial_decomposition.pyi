from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import (
    TriggerType,
    TaskTemplate as TaskTemplate,
)

_UNMATCHED_CUE_PREVIEW_COUNT: int

@dataclass(frozen=True, slots=True)
class DecomposedTrials:
    cumulative_distances: NDArray[np.float64]
    trial_names: tuple[str, ...]
    trigger_types: tuple[TriggerType, ...]

class CachedMotifDecomposer:
    _cached_motifs: list[NDArray[np.uint8]] | None
    _cached_flat_data: tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]] | None
    _cached_distances: NDArray[np.float32] | None
    def __init__(self) -> None: ...
    def prepare_motif_data(
        self, trial_motifs: list[NDArray[np.uint8]], trial_distances: list[float]
    ) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]: ...

def decompose_cue_sequence(
    cue_sequence: NDArray[np.uint8], task_template: TaskTemplate, motif_decomposer: CachedMotifDecomposer
) -> DecomposedTrials: ...
def _decompose_sequence_numba_flat(
    cue_sequence: NDArray[np.uint8],
    motifs_flat: NDArray[np.uint8],
    motif_starts: NDArray[np.int32],
    motif_lengths: NDArray[np.int32],
    motif_indices: NDArray[np.int32],
    max_trials: int,
) -> tuple[NDArray[np.int32], int, int]: ...
