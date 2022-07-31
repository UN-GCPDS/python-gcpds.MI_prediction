import numpy as np
import mne
import pandas as pd
from braindecode.preprocessing.windowers import _compute_window_inds, _check_windowing_arguments, WindowsDataset
from braindecode.datasets import BaseConcatDataset
from joblib import Parallel, delayed

def create_windows(win = 1, start_offset = 0, end_offset = 0, duration = 4, overlap = 0.0):
    st_offsets = []
    ed_offsets = []

    cont = start_offset + win
    while start_offset + win <= duration + end_offset:
        print(start_offset,cont-duration)
        st_offsets.append(start_offset)
        ed_offsets.append(cont-duration)

        start_offset += win*(1-overlap)
        cont += win*(1-overlap)

    return st_offsets, ed_offsets

def create_windows_from_events(
        concat_ds, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=None, window_stride_samples=None,
        drop_last_window=False, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None,
        on_missing='error', accepted_bads_ratio=0.0, n_jobs=1):

    _check_windowing_arguments(
        trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples, window_stride_samples)

    infer_window_size_stride = window_size_samples is None

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_events)(
            ds, infer_window_size_stride,
            trial_start_offset_samples, trial_stop_offset_samples,
            window_size_samples, window_stride_samples, drop_last_window,
            preload, drop_bad_windows, picks, reject, flat,
            on_missing, accepted_bads_ratio) for ds in concat_ds.datasets)

    return BaseConcatDataset(list_of_windows_ds)

def _create_windows_from_events(
        ds, infer_window_size_stride,
        trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples=None, window_stride_samples=None,
        drop_last_window=False, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None,
        on_missing='error', accepted_bads_ratio=0.0):

    events_id = None

    duration = int(ds.raw.n_times/ds.raw.info["sfreq"])
    onsets = np.array([0])
    stops = onsets+np.array([int(duration*ds.raw.info["sfreq"])])

    last_samp = ds.raw.first_samp + ds.raw.n_times
    if stops[-1] + trial_stop_offset_samples > last_samp:
        raise ValueError(
            '"trial_stop_offset_samples" too large. Stop of last trial '
            f'({stops[-1]}) + "trial_stop_offset_samples" '
            f'({trial_stop_offset_samples}) must be smaller than length of'
            f' recording ({len(ds)}).')

    window_size_samples = stops[0] + trial_stop_offset_samples - (onsets[0] + trial_start_offset_samples)
    window_stride_samples = window_size_samples

    i_trials, i_window_in_trials, starts, stops = _compute_window_inds(onsets, stops, trial_start_offset_samples,
        trial_stop_offset_samples, window_size_samples, window_stride_samples, drop_last_window,
        accepted_bads_ratio)
        
    description = -1
    events = [[start, window_size_samples, description]
                for i_start, start in enumerate(starts)]

    events = np.array(events)

    description = events[:, -1]

    metadata = pd.DataFrame({
        'i_window_in_trial': i_window_in_trials,
        'i_start_in_trial': starts,
        'i_stop_in_trial': stops,
        'target': description})

    mne_epochs = mne.Epochs(
        ds.raw, events, events_id, baseline=None, tmin=0,
        tmax=(window_size_samples - 1) / ds.raw.info["sfreq"],
        metadata=metadata,preload=preload, picks=picks, reject=reject,
        flat=flat, on_missing=on_missing)

    if drop_bad_windows:
            mne_epochs.drop_bad()

    windows_ds = WindowsDataset(mne_epochs, ds.description)
    return windows_ds
