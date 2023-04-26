import numpy as np

def resolve_monotone(Xs, Ys):
    idx = np.argsort(Xs)
    Xs_tmp = np.array(Xs)[idx]
    Ys_tmp = np.array(Ys)[idx]
    Xs_out = []
    Ys_out = []
    vx = 0
    vys = []
    for x, y in zip(Xs_tmp, Ys_tmp):
        if vys == []:
            vx = x
            vys.append(y)
        elif x == vx:
            vys.append(y)
        else:
            Xs_out.append(vx)
            Ys_out.append(sum(vys)/len(vys))
            vx = x
            vys = [y]
    if vys != []:
        Xs_out.append(vx)
        Ys_out.append(sum(vys)/len(vys))
    return np.array(Xs_out), np.array(Ys_out)

def running_average(Xs, Ys, w):
    Xs_out = []
    Ys_out = []
    for i in range(len(Xs) - w + 1):
        Xs_out.append(np.mean(Xs[i: i + w]))
        Ys_out.append(np.mean(Ys[i: i + w]))
    return np.array(Xs_out), np.array(Ys_out)