import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
np.seterr(divide='ignore', over='raise')

def u(r, li):
    epsilon = li[0]
    sigma = li[1]
    return 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

def grad_u(r, li):
    epsilon = li[0]
    sigma = li[1]
    return np.array([4*((sigma/r)**12 - (sigma/r)**6),
                     24*epsilon*(2*sigma**11/r**12 - sigma**5/r**6)])

def g(r, li):
    return np.exp(-u(r, li))

def optimize(li, r, gt, a):
    MAX_ITERATIONS = 1000
    iter_num = 0
    li_all = [[], []]
    lerrors = []
    iter_nums = []
    rs = []
    us = []
    gs = []

    while True:
        iter_nums.append(iter_num)
        if iter_num > MAX_ITERATIONS:
            raise RuntimeError('Could not find solution within ' + str(MAX_ITERATIONS) + ' iterations')
        
        lupdate = np.trapz((r**2)*(g(r, li) - gt)*grad_u(r, li), x=r)
        lerror = np.trapz((r*(g(r, li) - gt))**2, x=r)
        
        li_all[0].append(li[0])
        li_all[1].append(li[1])
        if iter_num%5 == 0:
            rs.append(r)
            us.append(u(r, li))
            gs.append(g(r, li))
        lerrors.append(lerror)
        
        if lerror < 0.001:
            print('Solution found in ' + str(iter_num + 1) + ' iterations')
            break
        elif np.all(abs(a*lupdate)/abs(li)) < 0.05:
            raise ValueError('Converged to wrong minimum')
        else:
            li += a*lupdate
            iter_num += 1
   
    result_plots(r, gt, iter_nums, lerrors, li_all, us, gs, rs)
    return li

def result_plots(r, gt, iter_nums, lerrors, li_all, us, gs, rs):
    plt.figure(figsize=(20,10))
    err = plt.subplot(221)
    params = plt.subplot(222)
    ucont = plt.subplot(223)
    gcont = plt.subplot(224)

    err.scatter(iter_nums, lerrors, c='k', marker = '.')
    err.hlines(0.001, 0, iter_nums[-1], colors='r', label=r'$\gamma$ = 0.001')

    x = np.linspace(0.5, 1.5, num=40)
    y = np.linspace(0.5, 1, num=10)
    vects = [[], []]
    for i in x:
        for j in y:
            xi = [i, j]
            gradS = np.trapz((r**2)*(g(r, xi) - gt)*grad_u(r, xi), x=r)
            gradS /= np.linalg.norm(gradS)
            vects[0].append(gradS[0])
            vects[1].append(gradS[1])          
    u, v = vects
    x, y = np.meshgrid(x, y, indexing='ij')
    params.quiver(x, y, u, v, angles='xy')
    params.scatter(li_all[0], li_all[1], marker='.', c=np.arange(len(iter_nums)), cmap='plasma')

    colors = mpl.cm.plasma(np.linspace(0, 1, len(rs)))
    for i, (r, ui, gi) in enumerate(zip(rs, us, gs)):
        ucont.plot(r, ui, c=colors[i]) 
        gcont.plot(r, gi, c=colors[i])
    ucont.plot(r, -np.log(gt), c='g', label='$u_{target}(r)$')
    gcont.plot(r, gt, c='g', label='$g_{target}(r)$')

    err.set_xlabel('i')
    err.set_ylabel('Error')
    err.set_title('Error Evolution')
    err.legend()
    params.set_xlabel(r'$\epsilon$')
    params.set_ylabel(r'$\sigma$')
    params.set_title('Design Evolution in the Gradient Field')
    ucont.set_xlabel('r')
    ucont.set_ylabel('u(r)')
    ucont.set_xlim(0.4, 1.25)
    ucont.set_title('u(r) Contour Evolution')
    ucont.set_ylim(-2.0, 5.0)
    ucont.legend()
    gcont.set_xlabel('r')
    gcont.set_ylabel('g(r)')
    gcont.set_xlim(right=2.0)
    gcont.set_title('g(r) Contour Evolution')
    gcont.legend()

    plt.savefig('optim_figs.png')

def main():
    target_data = np.loadtxt('g_target.txt')
    r = target_data[0]
    gt = target_data[1]
    a = 0.01
    li = np.array([1.0, 0.5])
    print(optimize(li, r, gt , a))

if __name__ == '__main__':
    main()