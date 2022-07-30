import numpy as np
import pandas as pd

def get_scores_cv(cv):
    results = cv.cv_results_
    mu = results['mean_test_score']
    sigma = results['std_test_score']
    return mu[np.argmax(sigma)], sigma[np.argmax(sigma)]

def get_scores(acc,targ):
    acc_dict = {}
    acc_dict['subject'] = []
    acc_dict['window'] = []
    acc_dict['accuracy'] = []
    acc_dict['std'] = []

    for s in acc.keys():
        for win in acc[s].keys():
            acc_dict['subject'].append(int(s))
            acc_dict['window'].append(win)
            acc_dict['accuracy'].append(acc[s][win]['acc_'+targ])
            if targ == 'train':
                acc_dict['std'].append(acc[s][win]['std_'+targ])
            elif targ == 'test':
                acc_dict['std'].append(None)

    acc_DF = pd.DataFrame.from_dict(acc_dict)
    return acc_DF