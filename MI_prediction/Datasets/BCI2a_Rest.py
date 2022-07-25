import numpy as np
from moabb.datasets.bnci import data_path
from moabb.datasets.base import BaseDataset
from scipy.io import loadmat
from mne.channels import make_standard_montage
from mne import create_info
from mne.io import RawArray
from mne.utils import verbose

BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"

@verbose
def _convert_run(run, ch_names=None, ch_types=None, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}
    n_chan = run.X.shape[1]
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    if not ch_names:
        ch_names = ["EEG%d" % ch for ch in range(1, n_chan + 1)]
        montage = None  # no montage

    if not ch_types:
        ch_types = ["eeg"] * n_chan

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)
    return raw

def _convert_mi(filename, ch_names, ch_types):
    runs = []
    event_id = {}
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    if isinstance(data["data"], np.ndarray):
        run_array = data["data"]
    else:
        run_array = [data["data"]]

    for run in run_array:
        if len(run.y)==0:
            raw = _convert_run(run, ch_names, ch_types, None)
            runs.append(raw)
    return runs

@verbose
def _load_data_001_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 001-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    # fmt: off
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
        "EOG1", "EOG2", "EOG3",
    ]
    # fmt: on
    ch_types = ["eeg"] * 22 + ["eog"] * 3

    sessions = {}
    for r in ["T", "E"]:
        url = "{u}001-2014/A{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        runs = _convert_mi(filename[0], ch_names, ch_types)
        # FIXME: deal with run with no event (1:3) and name them
        sessions["session_%s" % r] = {"run_%d" % ii: run for ii, run in enumerate(runs)}
    return sessions

@verbose
def load_data(
    subject,
    dataset="001-2014",
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None):  # noqa: D301
    
    dataset_list = {
        "001-2014": _load_data_001_2014,
    }

    baseurl_list = {
        "001-2014": BNCI_URL,}

    if dataset not in dataset_list.keys():
        raise ValueError(
            "Dataset '%s' is not a valid BNCI dataset ID. "
            "Valid dataset are %s." % (dataset, ", ".join(dataset_list.keys()))
        )

    return dataset_list[dataset](
        subject, path, force_update, update_path, base_url, verbose
    )

class MNEBNCI(BaseDataset):
    """Base BNCI dataset"""

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = load_data(subject=subject, dataset=self.code, verbose=False)
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return load_data(
            subject=subject,
            dataset=self.code,
            verbose=verbose,
            update_path=update_path,
            path=path,
            force_update=force_update,
        )

class BNCI2014001_Rest(MNEBNCI):
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events={"rest": 1},
            code="001-2014",
            interval=[0, 2],
            paradigm="imagery",
            doi="10.3389/fnins.2012.00055",
        )