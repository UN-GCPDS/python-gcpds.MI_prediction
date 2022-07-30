import numpy as np

def get_scores_cv(cv):
    results = cv.cv_results_
    mu = results['mean_test_score']
    sigma = results['std_test_score']
    return mu[np.argmax(sigma)], sigma[np.argmax(sigma)]