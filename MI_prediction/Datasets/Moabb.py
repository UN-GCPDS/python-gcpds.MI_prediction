import pandas as pd

from braindecode.datasets import BaseDataset as BD
from braindecode.datasets import BaseConcatDataset

def _fetch_and_unpack_moabb_data(dataset, subject_ids):
    data = dataset.get_data(subject_ids)
    raws, subject_ids, session_ids, run_ids = [], [], [], []
    for subj_id, subj_data in data.items():
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    description = pd.DataFrame({
        'subject': subject_ids,
        'session': session_ids,
        'run': run_ids
    })
    return raws, description

class MOABBDataset_Rest(BaseConcatDataset):
    def __init__(self, dataset, subject_ids, dataset_kwargs=None):
        raws, description = _fetch_and_unpack_moabb_data(dataset, subject_ids)
        all_base_ds = [BD(raw, row)
                       for raw, (_, row) in zip(raws, description.iterrows())]
        super().__init__(all_base_ds)