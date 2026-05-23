"""Provides utilities for decomposing a long Virtual Reality wall cue sequence into a sequence of trials.

The decomposition uses a greedy longest-match approach to identify trial motifs in the cue sequence received from
Unity. The CachedMotifDecomposer caches the flattened motif data between successive decompositions so that
re-arming Unity does not pay the flattening cost twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from numba import njit  # type: ignore[import-untyped]
import numpy as np
from ataraxis_base_utilities import console
from sollertia_shared_assets import TriggerType

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sollertia_shared_assets import TaskTemplate


@dataclass(slots=True)
class DecomposedTrials:
    """Stores the per-trial arrays derived from a decomposed Virtual Reality wall cue sequence.

    Notes:
        All arrays are aligned: index `i` describes the i-th trial in the decomposed sequence. The trial_names field
        is the join key the acquisition system uses to look up per-trial parameters from its experiment configuration.
    """

    cumulative_distances: NDArray[np.float64]
    """The cumulative distance, in centimeters, the animal must travel to reach the end of each decomposed trial."""
    trial_names: tuple[str, ...]
    """Name of each decomposed trial. Acquisition systems use these names to join the decomposed sequence with their
    own per-trial hardware parameters."""
    trigger_types: tuple[TriggerType, ...]
    """Stimulus trigger type for each decomposed trial. LICK indicates a positive (reward-zone) trial; OCCUPANCY
    indicates an aversive (occupancy-zone) trial."""


class CachedMotifDecomposer:
    """Caches the flattened trial cue sequence motif data between successive decomposition runs.

    Attributes:
        _cached_motifs: Stores the original trial motifs used for decomposition.
        _cached_flat_data: Stores the flattened motif data structure, optimized for numba-accelerated computations.
        _cached_distances: Stores the distances of each trial motif, in centimeters.
    """

    def __init__(self) -> None:
        self._cached_motifs: list[NDArray[np.uint8]] | None = None
        self._cached_flat_data: (
            tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]] | None
        ) = None
        self._cached_distances: NDArray[np.float32] | None = None

    def prepare_motif_data(
        self, trial_motifs: list[NDArray[np.uint8]], trial_distances: list[float]
    ) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
        """Prepares and caches the flattened motif data for faster cue sequence-to-trial decomposition.

        Args:
            trial_motifs: The trial motifs (wall cue sequences) to decompose.
            trial_distances: The trial motif distances, in centimeters.

        Returns:
            A tuple of five elements. The first element is the flattened array that stores all motifs. The second
            element is the array that stores the starting indices of each motif in the flattened array. The third
            element is the array that stores the length of each motif, in cues. The fourth element is the array
            that stores the original indices of motifs before sorting. The fifth element is the array of trial
            distances in centimeters.
        """
        # Returns cached data when the input motifs are unchanged across successive calls.
        if self._cached_motifs is not None and len(self._cached_motifs) == len(trial_motifs):
            all_equal = all(
                np.array_equal(cached, current)
                for cached, current in zip(self._cached_motifs, trial_motifs, strict=True)
            )
            if all_equal and self._cached_flat_data is not None and self._cached_distances is not None:
                # noinspection PyRedundantParentheses, PyTypeChecker
                return (*self._cached_flat_data, self._cached_distances)

        # Sorts motifs by length (longest first) so the greedy decomposer matches longer motifs before shorter ones.
        motif_data: list[tuple[int, NDArray[np.uint8], int]] = [
            (i, motif, len(motif)) for i, motif in enumerate(trial_motifs)
        ]
        motif_data.sort(key=lambda entry: entry[2], reverse=True)

        total_size: int = sum(len(motif) for motif in trial_motifs)
        num_motifs: int = len(trial_motifs)

        # noinspection PyTypeChecker
        motifs_flat: NDArray[np.uint8] = np.zeros(total_size, dtype=np.uint8)
        # noinspection PyTypeChecker
        motif_starts: NDArray[np.int32] = np.zeros(num_motifs, dtype=np.int32)
        # noinspection PyTypeChecker
        motif_lengths: NDArray[np.int32] = np.zeros(num_motifs, dtype=np.int32)
        # noinspection PyTypeChecker
        motif_indices: NDArray[np.int32] = np.zeros(num_motifs, dtype=np.int32)

        current_pos: int = 0
        for i, (orig_idx, motif, length) in enumerate(motif_data):
            motif_uint8 = motif.astype(np.uint8) if motif.dtype != np.uint8 else motif
            motifs_flat[current_pos : current_pos + length] = motif_uint8
            motif_starts[i] = current_pos
            motif_lengths[i] = length
            motif_indices[i] = orig_idx
            current_pos += length

        distances_array: NDArray[np.float32] = np.array(trial_distances, dtype=np.float32)

        self._cached_motifs = [motif.copy() for motif in trial_motifs]
        self._cached_flat_data = (motifs_flat, motif_starts, motif_lengths, motif_indices)
        self._cached_distances = distances_array

        # noinspection PyTypeChecker, PyRedundantParentheses
        return (*self._cached_flat_data, distances_array)


def decompose_cue_sequence(
    cue_sequence: NDArray[np.uint8],
    task_template: TaskTemplate,
    motif_decomposer: CachedMotifDecomposer,
) -> DecomposedTrials:
    """Decomposes a Virtual Reality cue sequence into per-trial distances, names, and trigger types.

    Notes:
        Uses a greedy longest-match approach to identify trial motifs in the cue sequence. The spatial trial layout
        (cue sequence, segment length) and the trigger type of each trial are sourced from the VR TaskTemplate.
        Acquisition-system-specific parameters (reward size, puff duration, etc.) are joined back by trial name on
        the acquisition system side.

    Args:
        cue_sequence: The Virtual Reality wall cue sequence to decompose, as a flat uint8 array.
        task_template: The VR TaskTemplate that provides the cue catalog, per-trial spatial cue sequences, and
            per-trial trigger types.
        motif_decomposer: The CachedMotifDecomposer instance used to flatten and cache the trial motif data.

    Returns:
        The DecomposedTrials instance with three aligned arrays: cumulative distances, trial names, and trigger types.

    Raises:
        RuntimeError: If the decomposer cannot match any trial motif at some position in the cue sequence.
    """
    cue_name_to_code = {cue.name: int(cue.code) for cue in task_template.cues}
    cue_name_to_length = {cue.name: float(cue.length_cm) for cue in task_template.cues}

    trial_motifs: list[NDArray[np.uint8]] = []
    trial_distances: list[float] = []
    trial_names_by_type: list[str] = []
    trigger_types_by_type: list[TriggerType] = []

    for trial_name, spatial_trial in task_template.trial_structures.items():
        trial_motifs.append(np.array([cue_name_to_code[name] for name in spatial_trial.cue_sequence], dtype=np.uint8))
        trial_distances.append(sum(cue_name_to_length[name] for name in spatial_trial.cue_sequence))
        trial_names_by_type.append(trial_name)
        trigger_types_by_type.append(
            spatial_trial.trigger_type
            if isinstance(spatial_trial.trigger_type, TriggerType)
            else TriggerType(spatial_trial.trigger_type)
        )

    motifs_flat, motif_starts, motif_lengths, motif_indices, distances_array = motif_decomposer.prepare_motif_data(
        trial_motifs, trial_distances
    )

    max_trials = len(cue_sequence) // min(len(motif) for motif in trial_motifs) + 1
    trial_indices_array, trial_count = _decompose_sequence_numba_flat(
        cue_sequence, motifs_flat, motif_starts, motif_lengths, motif_indices, max_trials
    )

    if trial_count == -1:
        failed_position = sum(len(trial_motifs[index]) for index in trial_indices_array[:max_trials] if index != 0)
        remaining_cues = cue_sequence[failed_position : failed_position + 20]
        message = (
            f"Unable to decompose the acquired session's Virtual Reality environment's cue sequence into a sequence "
            f"of trials. No trial motif matched the processed sequence at position {failed_position}. The next 20 "
            f"unmatched cues: {remaining_cues.tolist()}."
        )
        console.error(message=message, error=RuntimeError)

    sequence_indices = trial_indices_array[:trial_count]
    cumulative_distances = np.cumsum(distances_array[sequence_indices].astype(np.float64))
    trial_names = tuple(trial_names_by_type[index] for index in sequence_indices)
    trigger_types = tuple(trigger_types_by_type[index] for index in sequence_indices)

    return DecomposedTrials(
        cumulative_distances=cumulative_distances,
        trial_names=trial_names,
        trigger_types=trigger_types,
    )


@njit(cache=True)
def _decompose_sequence_numba_flat(
    cue_sequence: NDArray[np.uint8],
    motifs_flat: NDArray[np.uint8],
    motif_starts: NDArray[np.int32],
    motif_lengths: NDArray[np.int32],
    motif_indices: NDArray[np.int32],
    max_trials: int,
) -> tuple[NDArray[np.int32], int]:
    """Decomposes a long sequence of Virtual Reality wall cues into individual trial motifs.

    Notes:
        This worker function is used to speed up decomposition via numba acceleration.

    Args:
        cue_sequence: The full Virtual Reality environment cue sequence to decompose.
        motifs_flat: All trial type motifs supported by the acquired session, concatenated into a single 1D array.
        motif_starts: The starting index of each unique motif in the motifs_flat array.
        motif_lengths: The length of each unique motif in the motifs_flat array.
        motif_indices: Stores the original trial type motif indices before they are sorted to optimize the lookup
            speed.
        max_trials: The maximum number of trials that can make up the entire cue sequence.

    Returns:
        A tuple of two elements. The first element is the array of trials (trial-type indices) decoded from the cue
        sequence. The second element is the total number of trials extracted from the cue sequence.
    """
    trial_indices = np.zeros(max_trials, dtype=np.int32)
    trial_count = 0
    sequence_pos = 0
    sequence_length = len(cue_sequence)
    num_motifs = len(motif_lengths)

    while sequence_pos < sequence_length and trial_count < max_trials:
        motif_found = False

        for i in range(num_motifs):
            motif_length = motif_lengths[i]

            if sequence_pos + motif_length <= sequence_length:
                motif_start = motif_starts[i]

                match = True
                for j in range(motif_length):
                    if cue_sequence[sequence_pos + j] != motifs_flat[motif_start + j]:
                        match = False
                        break

                if match:
                    trial_indices[trial_count] = motif_indices[i]
                    trial_count += 1
                    sequence_pos += motif_length
                    motif_found = True
                    break

        if not motif_found:
            return trial_indices, -1

    return trial_indices[:trial_count], trial_count
