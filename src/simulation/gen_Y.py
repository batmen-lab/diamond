import numpy as np

def f_1(x):
    return np.pi ** (x[0] * x[1]) * np.sqrt(2 * x[2]) - \
           (np.arcsin(x[3])) + np.log(1 + x[2] + x[4]) - \
           (x[8] / (1 + x[9])) * np.sqrt(x[6] / (1 + x[7])) - \
           x[1] * x[6], \
           [{0, 1, 2}, {2, 4}, {6, 7, 8, 9}, {1, 6}], \
           {0, 1, 2, 3, 4, 6, 7, 8, 9}

def f_2(x):
    return np.pi ** (x[0] * x[1]) * np.sqrt(2 * np.abs(x[2])) - \
           (np.arcsin(0.5 * x[3])) + np.log(np.abs(x[2] + x[4]) + 1) - \
           (x[8] / (1 + np.abs(x[9]))) * np.sqrt(np.abs(x[6]) / (1 + np.abs(x[7]))) - \
           x[1] * x[6], \
           [{0, 1, 2}, {2, 4},{6, 7, 8, 9},{1, 6}], \
           {0, 1, 2, 3, 4, 6, 7, 8, 9}

def f_3(x):
    return np.exp(np.abs(x[0] - x[1])) + np.abs(x[1] * x[2]) - \
           ((x[2]) ** 2) ** np.absolute(x[3]) + np.log(x[3] ** 2 + \
           x[4] ** 2 + x[6] ** 2 + x[7] ** 2) + x[8] + 1. / (1 + x[9] ** 2), \
           [{0, 1}, {1, 2}, {2, 3}, {3, 4, 6, 7}], \
           {0, 1, 2, 3, 4, 6, 7, 8, 9}

def f_4(x):
    return np.exp(np.abs(x[0] - x[1])) + \
           np.abs(x[1] * x[2]) - ((x[2]) ** 2) ** np.absolute(x[3]) + \
           (x[0] * x[3]) ** 2 + np.log(x[3] ** 2 + x[4] ** 2 + x[6] ** 2 + x[7] ** 2) + \
           x[8] + 1. / (1 + x[9] ** 2), \
           [{0, 1}, {1, 2},{2, 3}, {0, 3},{3, 4, 6, 7}], \
           {0, 1, 2, 3, 4, 6, 7, 8, 9}

def f_5(x):
    return 1. / (1 + x[0] ** 2 + x[1] ** 2 + x[2] ** 2) + \
           np.sqrt(np.abs(x[3] + x[4])) + np.abs(x[5] + x[6]) + \
           x[7] * x[8] * x[9], \
           [{0, 1, 2}, {3, 4}, {5, 6}, {7, 8, 9}], \
           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_6(x):
    return np.exp(np.abs(x[0] * x[1]) + 1) \
           - np.exp(np.abs(x[2] + x[3]) + 1) \
           + np.cos(x[4] + x[5] - x[7]) + \
           np.sqrt(x[7] ** 2 + x[8] ** 2 + x[9] ** 2), \
           [{0, 1}, {2, 3}, {4, 5, 7}, {7, 8, 9}], \
           {0, 1, 2, 3, 4, 5, 7, 8, 9}

def f_7(x):
    return (np.arctan(x[0]) + np.arctan(x[1])) ** 2 + \
           np.max((x[2] * x[3] + x[5], np.zeros(x.shape[1])), 0) - \
           (1 + (x[3] * x[4] * x[5] * x[6] * x[7])) ** -2 + \
           (np.abs(x[6]) / (1 + np.abs(x[8]))) ** 5 + np.sum(x[:10], axis=0), \
           [{0, 1}, {2, 3, 5}, {3, 4, 5, 6, 7}, {6, 8}], \
           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_8(x):
    return x[0] * x[1] + 2 ** (x[2] + x[4] + x[5]) + \
           2 ** (x[2] + x[3] + x[4] + x[6]) + \
           np.sin(x[6] * np.sin(x[7] + x[8])) + \
           np.arccos(0.9 * x[9]), \
           [{0, 1}, {2, 4, 5}, {2, 3, 4, 6}, {6, 7, 8}], \
           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_9(x):
    return np.tanh(x[0] * x[1] + x[2] * x[3]) * np.sqrt(np.abs(x[4])) + \
           np.exp(x[4] + x[5]) + np.log((x[5] * x[6] * x[7]) ** 2 + 1) + \
           x[8] * x[9] + 1. / (1 + np.abs(x[9])), \
           [{0, 1, 2, 3, 4}, {4, 5}, {5, 6, 7},{8, 9}], \
           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_10(x):
    return np.sinh(x[0] + x[1]) + np.arccos(np.tanh(x[2] + x[4] + x[6])) + \
           np.cos(x[3] + x[4]) + 1. / np.cos(x[6] * x[8]), \
           [{0, 1}, {2, 4, 6}, {3, 4}, {6, 8}], \
           {0, 1, 2, 3, 4, 6, 8}

def generate_interaction_response(X, func_num):
    Y, interct_gt, margin_gt = globals()["f_{}".format(func_num)](X.transpose())
    Y = Y.reshape(len(Y),1).astype(np.float32)
    return Y, interct_gt, margin_gt
