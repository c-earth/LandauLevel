import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def poly(x, *ps):
    out = 0
    for i in range(len(ps)):
        out += ps[i] * x ** i
    return out

def dpoly(x, *ps):
    out = 0
    for i in range(len(ps)):
        if i == 0:
            continue
        else:
            out += i * ps[i] * x ** (i - 1)
    return out

def pieces_poly(xm, *bspss):
    x = xm[:-1]
    m = int(xm[-1])
    bs = np.sort(np.concatenate([[0, np.inf], bspss[:m-1]]))
    pss = np.array(bspss[m-1:]).reshape((m, -1))

    fnb = 0
    dfnb = 0
    out = np.zeros(x.shape)

    for i, (bi, ps, bf) in enumerate(zip(bs[:-1], pss, bs[1:])):
        if i == 0:
            out += (bi <= x) * (x < bf) * poly(x-bi, *ps)
            fnb = poly(bf-bi, *ps)
            dfnb = dpoly(bf-bi, *ps)
        else:
            out += (bi <= x) * (x < bf) * poly(x-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
            fnb = poly(bf-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
            dfnb = dpoly(bf-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
    
    return out

def subbg_poly(data, temperatures, n):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))
    out = dict()

    plt.figure()
    for temperature in temperatures:
        H = data[temperature][0]
        rho_xx = data[temperature][1]
        ps, _ = curve_fit(poly, H, rho_xx, p0 = np.ones(n), sigma = np.sqrt(rho_xx + 1) - 1)
        plt.plot(H, rho_xx, '-', color = colors[int(temperature)])
        plt.plot(H, poly(H, *ps), '--', color = colors[int(temperature)])
        out[temperature] = [H, rho_xx - poly(H, *ps)] 
    plt.show()
    return out

def subbg_pieces_poly(data, temperatures, m, n):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))
    out = dict()

    plt.figure()
    for temperature in temperatures:
        H = data[temperature][0]
        rho_xx = data[temperature][1]
        bspss, _ = curve_fit(pieces_poly, np.concatenate([H, [m]]), rho_xx, p0 = np.ones(m - 1 + m * n), sigma = np.sqrt(rho_xx + 1) - 1, maxfev = 20000)
        plt.plot(H, rho_xx, '-', color = colors[int(temperature)])
        plt.plot(H, pieces_poly(np.concatenate([H, [m]]), *bspss), '--', color = colors[int(temperature)])
        out[temperature] = [H, rho_xx - pieces_poly(np.concatenate([H, [m]]), *bspss)]
    plt.show()
    return out

def subbg_derivative(data, temperatures, n):
    out = dict()
    for temperature in temperatures:
        H = data[temperature][0]
        rho_xx = data[temperature][1]
        for i in range(1, n+1):
            rho_xx = (rho_xx[1:] - rho_xx[:-1]) / (H[1:] - H[-1])
            H = (H[1:] + H[-1])/2
        out[temperature] = [H, rho_xx]
    return out