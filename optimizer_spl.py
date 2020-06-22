import numpy as np
from scipy.interpolate import Akima1DInterpolator
import matplotlib as mpl
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', over='raise')

def u(r, m, li):
    return Akima1DInterpolator(m, li).__call__(r)

def g(r, m, li):
    return np.exp(-u(r, m, li))

def grad_uu(r, m, li):
    h = 0.001
    du = np.ndarray(shape=(len(m),len(r)))
    
    for i, j in enumerate(li):
        li_temp_high = np.copy(li)
        li_temp_low = np.copy(li)
        li_temp_high[i] += h
        li_temp_low[i] -= h
        
        u_i_high = u(r, m, li_temp_high)
        u_i_low = u(r, m, li_temp_low)
        du[i] = (u_i_high - u_i_low)/(2*h)
        
    return du

def grad_ut(r, m, li):
    h = 0.001
    du = np.ndarray(shape=(len(m),len(r)))

    for i, j in enumerate(li):
        ui_temp_high = np.copy(li)
        ui_temp_low = np.copy(li)
        li_temp_high = np.copy(li)
        li_temp_low = np.copy(li)
        li_temp_high[i] += h
        li_temp_low[i] -= h
        
        for n, (p, q) in enumerate(zip(li_temp_high, li_temp_low)):
            ui_temp_high[n] = ui_temp_high[-1] + np.sum(li_temp_high[n:-1])
            ui_temp_low[n] = ui_temp_low[-1] + np.sum(li_temp_low[n:-1])
        
        u_i_high = u(r, m, ui_temp_high)
        u_i_low = u(r, m, ui_temp_low)
        du[i] = (u_i_high - u_i_low)/(2*h)
   
    return du

def optimize_u(li, r, gt, a, step_tol, m, uplot, gplot): 
    MAX_ITERATIONS = 1000
    iter_num = 0
    rs = []
    gs = []
    
    while True:
        if iter_num > MAX_ITERATIONS:
            raise RuntimeError('Could not find solution within ' + str(MAX_ITERATIONS) + ' iterations')
        
        lupdate = np.trapz((r**2)*(g(r, m, li) - gt)*grad_uu(r, m, li), x=r)
        step_error = abs(a*lupdate)/abs(li)
        if iter_num >= 1:
            step_error = step_error[np.isfinite(step_error)]

        rs.append(r)
        gs.append(g(r, m, li))
        
        if np.all(step_error < step_tol):
            print('Solution found in ' + str(iter_num) + ' iterations')
            break
        else:
            li += a*lupdate
            iter_num += 1

    result_plots_u(r, gt, rs, gs, uplot, gplot)
    return li

def optimize_theta(li, r, gt, a, step_tol, m, uplot, gplot, constr_min=-np.inf): 
    MAX_ITERATIONS = 2500
    iter_num = 0
    ui = np.copy(li)
    rs = []
    gs = []

    while True:
        if iter_num > MAX_ITERATIONS:
            raise RuntimeError('Could not find solution within ' + str(MAX_ITERATIONS) + ' iterations')
 
        for i, j in enumerate(li):
            ui[i] = ui[-1] + np.sum(li[i:-1])
        lupdate = np.trapz((r**2)*(g(r, m, ui) - gt)*grad_ut(r, m, li), x=r)
        step_error = abs(a*lupdate)/abs(li)
        if iter_num >= 1:
            step_error = step_error[np.isfinite(step_error)]

        rs.append(r)
        gs.append(g(r, m, ui))

        if np.all(step_error < step_tol):
            print('Solution found in ' + str(iter_num) + ' iterations')
            break
        else:
            li += a*lupdate
            li = np.maximum(constr_min, li)
            iter_num += 1

    result_plots_theta(r, gt, rs, gs, uplot, gplot, constr_min)
    return li

def result_plots_u(r, gt, rs, gs, uplot, gplot):  
    colors = mpl.cm.plasma(np.linspace(0, 1, len(rs)))

    for i, (r, gi) in enumerate(zip(rs, gs)):
        uplot.plot(r, -np.log(gi), c=colors[i]) 
        gplot.plot(r, gi, c=colors[i])
    uplot.plot(r, -np.log(gt), c='g', label='$u_{target}(r)$')
    gplot.plot(r, gt, c='g', label='$g_{target}(r)$')
    
    uplot.set_xlabel('r')
    uplot.set_ylabel('u(r)')
    uplot.set_xlim(0.75, 1.25)
    uplot.set_title('u(r) Contour Evolution - ' + r'$\lambda_{j} = u_{j}$')
    uplot.set_ylim(-2.0, 5.0)
    uplot.legend(loc=1)
    gplot.set_xlabel('r')
    gplot.set_ylabel('g(r)')
    gplot.set_xlim(0.5, 2.0)
    gplot.set_title('g(r) Contour Evolution - ' + r'$\lambda_{j} = u_{j}$')
    gplot.legend(loc=1)

def result_plots_theta(r, gt, rs, gs, uplot, gplot, constr_min):  
    colors = mpl.cm.plasma(np.linspace(0, 1, len(rs)))

    for i, (r, gi) in enumerate(zip(rs, gs)):
        uplot.plot(r, -np.log(gi), c=colors[i]) 
        gplot.plot(r, gi, c=colors[i])
    uplot.plot(r, -np.log(gt), c='g', label='$u_{target}(r)$')
    gplot.plot(r, gt, c='g', label='$g_{target}(r)$')
    
    uplot.set_xlabel('r')
    uplot.set_ylabel('u(r)')
    uplot.set_xlim(0.75, 1.25)
    uplot.set_ylim(-2.0, 5.0)
    uplot.legend(loc=1)
    gplot.set_xlabel('r')
    gplot.set_ylabel('g(r)')
    gplot.set_xlim(0.5, 2.0)
    gplot.legend(loc=1)
    if constr_min != -np.inf:
        uplot.set_title('u(r) Contour Evolution - ' + r'$\lambda_{j} = \theta_{j}$' + ' - ' + r'$\theta_{j} ≥ $' + str(constr_min))
        gplot.set_title('g(r) Contour Evolution - ' + r'$\lambda_{j} = \theta_{j}$' + ' - ' + r'$\theta_{j} ≥ $' + str(constr_min))
    else:
        uplot.set_title('u(r) Contour Evolution - ' + r'$\lambda_{j} = \theta_{j}$')
        gplot.set_title('g(r) Contour Evolution - ' + r'$\lambda_{j} = \theta_{j}$')

def main():
    target_data = np.loadtxt('g_target.txt')
    r = target_data[0]
    gt = target_data[1]
    plt.figure(figsize=(20,20))
    
    m_len = 50
    m = np.linspace(r[0], r[-1], m_len)
    
    a = 0.05
    step_tol = 0.005
    li = np.zeros_like(m)
    print(optimize_u(li, r, gt, a, step_tol, m, plt.subplot(321), plt.subplot(322)))
    
    a = 0.00375
    step_tol = 0.0005
    li = np.zeros_like(m)
    print(optimize_theta(li, r, gt, a, step_tol, m, plt.subplot(323), plt.subplot(324)))
    
    a = 40
    step_tol = 0.005
    li = np.zeros_like(m)
    print(optimize_theta(li, r, gt, a, step_tol, m, plt.subplot(325), plt.subplot(326), constr_min=0))
    
    plt.savefig('optim_spl_figs.png')
    
if __name__ == '__main__':
    main()
