from braindecode.preprocessing.preprocess import Preprocessor
import copy

def filterbank_preprocessor(freqs):
    FB = {}
    for freq in freqs:
        print('band to filter: {} Hz'.format(freq))
        FB[str(freq[0])+'_'+str(freq[1])] = Preprocessor('filter', l_freq=freq[0], h_freq=freq[1])
    return FB

def filterbank(ds, preprocess = [], filters = [], standarization = [], channels_prep = []):
    ds_filt = []
    for i in filters.keys():
        d_tmp = copy.deepcopy(ds)
        preprocessors = preprocess + channels_prep + [filters[i]] + standarization
        d_tmp.preprocess_data(preprocessors=preprocessors)
        ds_filt.append(d_tmp)
    return ds_filt
