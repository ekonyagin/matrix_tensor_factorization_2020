import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.animation as animation
import matplotlib
from sklearn.linear_model import LinearRegression
import pickle 

def load(filename):
    if filename[-3:] == "npy":
        return np.load(filename)
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data

def linregress(X, y):
    lr = LinearRegression().fit(X, y)
    print("%.3f x + %.3f y + %.3f > z" % (lr.coef_[0][0], lr.coef_[0][1], lr.intercept_))
    return lr.coef_, lr.intercept_


def __simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def __dominates(row_, candidateRow):
    row = row_[:3]
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)  





def pareto_and_dominated(points):
    if type(points) is not list:
        points = points.tolist()
    paretoPoints, dominatedPoints = __simple_cull(points, __dominates)
    return np.array(list(paretoPoints)), np.array(list(dominatedPoints))


def plot_pareto_frontier(points, labels=["model accuracy", "compression rate", "speed-up"]):
    paretoPoints_, dominatedPoints_ = pareto_and_dominated(points)
    paretoPoints, dominatedPoints = paretoPoints_.T[:3].T, dominatedPoints_.T[:3].T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))
    # print(pp.shape,dp.shape)
    ax.scatter(dp[:,0],dp[:,1],dp[:,2], label="dominated")
    ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red', label="Pareto optimal")
    triang = mtri.Triangulation(pp[:,0],pp[:,1])
    ax.plot_trisurf(triang,pp[:,2], color=(1,0,0,0.4), edgecolor='Black')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.legend()
    plt.show()


def approximate_upper_bound(points):
    paretoPoints, _ = pareto_and_dominated(points)
    return linregress(paretoPoints.T[[1, 2]].T, paretoPoints.T[[0]].T)


def plot_optimal_configurations(points, metrics, labels=["", "", ""], plot_triangulation=False):
    paretoPoints_, _ = pareto_and_dominated(points)
    paretoPoints = paretoPoints_.T[metrics].T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pp = np.array(list(paretoPoints))
    # print(pp.shape,dp.shape)
    ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
    if plot_triangulation:
        triang = mtri.Triangulation(pp[:,0],pp[:,1])
        ax.plot_trisurf(triang,pp[:,2], color=(0,1,0,0.4), edgecolor='Black')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.show()    
    

def scatter_binary(points, metrics=[0, 1], labels=["", ""]):
    paretoPoints_, dominatedPoints_ = pareto_and_dominated(points)
    fig, ax = plt.subplots()
    paretoPoints, dominatedPoints = paretoPoints_.T[metrics].T, dominatedPoints_.T[metrics].T
    ax.scatter(dominatedPoints.T[0], dominatedPoints.T[1], label="dominated")
    ax.scatter(paretoPoints.T[0], paretoPoints.T[1], color='red', label="Pareto optimal")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    plt.legend()
    plt.show()






#______________________________________________________________________________________


def plot_3d_tensor(x, y, z, Z, labels=["r_1", "r_2", "relative error"], fps=1, zlim=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    if zlim is not None:
        ax.set_zlim(*zlim)
    X, Y = np.meshgrid(x, y)
    def update(idx):
        if update.wframe:
            ax.collections.remove(update.wframe)
        ax.set_title("r = %d" % z[idx])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        update.wframe = ax.plot_wireframe(X, Y, Z[idx], rstride=1, cstride=1, color='k', linewidth=0.5)
    update.wframe = None
    ani = animation.FuncAnimation(fig, update, Z.shape[0], interval=1000/fps)
    fn = 'plot_wireframe_funcanimation'
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)


rs_ = [100, 120, 150, 180, 210, 240, 250, 280, 300, 330, 380, 400]
rs = [100, 120, 150, 180, 240, 250]
r1s = [30, 60, 90]
r2s = [30, 60, 90]

def animate_relative_errors(filenames):    
    data = dict()
    for filename in filenames:
        f = open(filename, "rb")
        data.update(pickle.load(f))
        print(list(data))
        f.close()
    Z = np.array([[[data["%d %d" % (r1, r2)][r] for r2 in r2s] for r1 in r1s] for r in rs])
    plot_3d_tensor(r1s, r2s, rs_, Z)

def animate_speed(filename, reference, zlim):
    Z = reference / np.transpose(load(filename), (2, 0, 1))
    plot_3d_tensor(r1s, r2s, rs, Z, labels=["r1", "r2", "Speed-up"], zlim=zlim)

# animate_relative_errors(["data/decomp_%d.pickle" % n for n in [60, 90]])

# animate_speed("data/time_gpu.npy", 0.00175, (1.55, 3.15))

def CompressionRate(r1,r2,R):
    s = 9*r1*r2 + 512*(r1+r2) + 9*R + R*(r1+r2)
    init = 512*512*9.
    return init/s

Zcr = np.array([[[ CompressionRate(r1, r2, r) for r2 in r2s] for r1 in r1s] for r in rs])
plot_3d_tensor(r1s, r2s, rs, Zcr, labels=["r1", "r2", "Compression rate"], zlim=(14, 55))

if __name__ == "__main__":
    points = np.array([[random.randint(70,100) for i in range(3)] for j in range(500)])
    #plot_pareto_frontier(points)
    #scatter_binary(points)
    # plot_optimal_configurations(points, metrics, labels=["", "", ""], plot_triangulation=False)
    #approximate_upper_bound(points)
    x = np.arange(3)
