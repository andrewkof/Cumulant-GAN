"""
@author: andrewkof

Auxiliary functions.
"""
import os
import csv
import glob
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
from scipy.stats import multivariate_normal
from matplotlib import animation

def get_data(name):                                # Input: csv file (2 columns, unknown rows)
    x,y = [],[]                                    # Output: lists x,y representing generated data.
    with open(name) as f:
        for row in f:
            x.append(float(row.split(',')[0]))
            y.append(float(row.split(',')[1]))
    return x, y

def get_data_name(name):                            # Get dataset via parser name.
    if name == 'gmm8':
        return 'toy_example_gmm8.csv'
    elif name == 'tmm6':
        return 'toy_example_tmm6.csv'
    return 'swiss_roll_2d_with_labels.csv'

def get_divergence_name(beta, gamma):

    if beta == 0.0 and gamma == 0.0:
        return 'Wasserstein Metric'

    elif beta == 0.0 and gamma == 1.0:
        return 'Kullback-Leibler Divergence'

    elif beta == 1.0 and gamma == 0.0:
        return 'Reverse Kullback-Leibler Divergence'

    elif beta == 0.5 and gamma == 0.5:
        return '-4log(1-Hellinger^2)'

    return 'beta={}_gamma={}'.format(beta,gamma)

def save_data(data, divergence_name, beta, gamma, iter):         # Save data to csv file
    if not os.path.exists(divergence_name):
        os.makedirs(divergence_name)
    data_ = data.numpy()

    with open(divergence_name + '/data/' + 'cumgan_samples_beta_' + str(beta)+'_gamma_' + str(gamma)+ '_iteration_' + str(iter) + '.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in data_:
            writer.writerow(val)

# Partitioning to save specific datasets during training process.
def partition(name):
    if name == 'gmm8' or name == 'tmm6':    # Partitioning for gmm8 and tmm6 datasets.
        return [i for i in range(0,1001,50)] + [i for i in range(1100,5001,100)] + [i for i in range(5200,10001,200)]

    return [i for i in range(0,100001,1000)] # Swiss Roll partition

def get_csvs(name):
    directory = name + '/data/'
    csv_names = filter(os.path.isfile,glob.glob(directory + '*'))
    csv_names = sorted(csv_names, key=os.path.getmtime)
    frames = len(csv_names)
    return csv_names, frames

def generate_and_save_plots(divergence_name, data_name, beta, gamma, generated, iter):
    # create directory to save data and plots
    if not os.path.exists(divergence_name + '/plots'):
        os.makedirs(divergence_name + '/plots')
        os.makedirs(divergence_name + '/data')

    save_data(generated, divergence_name, beta, gamma, iter)

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.plot(generated.numpy()[:,0], generated.numpy()[:,1], 'ro',markersize=2)
    plt.grid()
    plt.title('Iteration = {}'.format(iter))
    fig.suptitle('{}'.format(divergence_name))
    if data_name == 'swissroll':
        plt.axis([-20, 20, -20, 20])
    else:
        plt.axis([-6, 6, -6, 6])

    plt.savefig(divergence_name +'/plots/plots{}.png'.format(str(iter).zfill(3)), bbox_inches='tight')
    plt.close(fig)

# Create real data (6 student's T distributions)
def tmm6_contour():
    e = 3*np.array([1,0]).T
    d = len(e)
    K = 6
    theta = 2*np.pi/K
    x, y = np.mgrid[-6:6.1:.1, -6:6.1:.1]
    pos = np.dstack((x, y))

    sigma = 1.0
    Sigma = sigma*np.eye(d)

    rtt_mtx = np.array([[np.cos((1-1)*theta), -np.sin((1-1)*theta)], [np.sin((1-1)*theta),np.cos((1-1)*theta)]])
    mn_vec = np.matmul(rtt_mtx,e)
    rv = 1/K* multivariate_t(loc = mn_vec, df=3,shape=Sigma).pdf(pos)

    for i in range(2,K+1):

        rtt_mtx = np.array([[np.cos((i-1)*theta), -np.sin((i-1)*theta)],[np.sin((i-1)*theta), np.cos((i-1)*theta)]])
        mn_vec = np.matmul(rtt_mtx,e)

        z2 = multivariate_t(loc = mn_vec,df =3, shape=Sigma)
        rv += 1/K * z2.pdf(pos)
    return x, y, rv

# Create 8 equiprobable and equidistant-from-the-origin Gaussians
def gmm8_contourf():
    K = 8
    e = 3*np.array([1,0]).T
    d = len(e)
    sigma = 0.5
    Sigma = sigma*np.eye(d)
    theta = 2*np.pi/K
    x, y = np.mgrid[-6:6.1:.1, -6:6.1:.1]
    pos = np.dstack((x, y))

    rtt_mtx = np.array([[np.cos((1-1)*theta), -np.sin((1-1)*theta)], [np.sin((1-1)*theta),np.cos((1-1)*theta)]])
    mn_vec = np.matmul(rtt_mtx,e)
    rv = multivariate_normal(mean = mn_vec,cov = Sigma).pdf(pos)

    for i in range(2,K+1):
        rtt_mtx = np.array([[np.cos((i-1)*theta), -np.sin((i-1)*theta)], [np.sin((i-1)*theta),np.cos((i-1)*theta)]])
        mn_vec = np.matmul(rtt_mtx,e)
        z2 = multivariate_normal(mean = mn_vec, cov = Sigma)
        rv += z2.pdf(pos)
    cs = plt.contour(x, y, rv, linewidths = 2)
    return cs

# Create Swiss Roll real data
def swissroll_contour():
    e = (3*np.pi/2)*np.array([0,-1])
    d = len(e)

    sigma = 0.1
    Sigma = sigma*np.eye(d)

    K = 1000
    theta = 3*np.pi/K
    x, y = np.mgrid[-20:20.1:0.1, -20:20.1:0.1]
    pos = np.dstack((x, y))

    rtt_mtx = (1+2*1/K) * np.array([[np.cos((0-2)*theta), -np.sin((0-2)*theta)], [np.sin((0-2)*theta), np.cos((0-2)*theta)]])
    mn_vec = np.matmul(rtt_mtx,e)
    rv = 1/K * multivariate_normal(mean = mn_vec,cov = Sigma).pdf(pos)

    for i in range(1,K+1):
        rtt_mtx = (1+2*(i+1)/K) * np.array([[np.cos((i-2)*theta), -np.sin((i-2)*theta)], [np.sin((i-2)*theta), np.cos((i-2)*theta)]])
        mn_vec = np.matmul(rtt_mtx,e)
        z2 = multivariate_normal(mean = mn_vec,cov = Sigma)
        rv += (1+np.log(i+1)**2)/K * z2.pdf(pos)

    plt.xlim(-20,20)
    plt.ylim(-20,20)
    return x, y, rv

def create_gmm8_clip(data_name):

    csv_names, frames = get_csvs(data_name)
    iterations = partition('gmm8')

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.grid()
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    def animate(i):
        x, y = get_data(csv_names[i])
        ax.clear()
        plt.grid()
        gmm8_contourf()
        plt.plot(x, y, 'ro', markersize=1.5)
        plt.title('Iteration = {}'.format(iterations[i]), fontweight="bold", fontsize=15)
        plt.legend(['Generated data'], loc = 'upper right')
        plt.axis([-6, 6, -6, 6])
        plt.pause(0.01)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=20)
    anim.save(data_name + '.gif', writer= animation.PillowWriter(fps=7))

def create_tmm6_clip(name):

    csv_names, frames = get_csvs(name)
    iterations = partition('tmm6')

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.grid()
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    x_, y_, rv_ = tmm6_contour()
    def animate(i):
        x, y = get_data(csv_names[i])
        ax.clear()
        plt.grid()
        plt.contour(x_,y_,rv_, linewidths = 2)
        plt.plot(x,y,'ro',markersize=1.5)                                     # Fake data
        plt.title('Iteration = {}'.format(iterations[i]), fontweight="bold", fontsize=15)
        plt.legend(['Generated data'], loc = 'upper right')
        plt.axis([-6, 6, -6, 6])
        plt.pause(0.01)

    anim = animation.FuncAnimation(fig, animate, frames=frames,interval=20)
    anim.save(name + '.gif', writer= animation.PillowWriter(fps=7))

def create_swiss_roll_clip(name):

    csv_names, frames = get_csvs(name)
    iterations = partition('swissroll')

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.grid()
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    x1, y1, rv = swissroll_contour()

    def animate(i):
        x, y = get_data(csv_names[i])
        ax.clear()
        plt.grid()
        plt.contour(x1 ,y1, rv, extend = 'min')
        plt.plot(x,y,'ro',markersize=1.5)
        plt.title('Iteration = {}'.format(iterations[i]), fontweight="bold", fontsize=15)
        plt.legend(['Generated data'], loc = 'upper right')
        plt.axis([-20, 20, -20, 20])
        plt.pause(0.01)

    anim = animation.FuncAnimation(fig, animate, frames=frames ,interval=20)
    anim.save(name + '.gif', writer= animation.PillowWriter(fps=7))