import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from sklearn.linear_model import LinearRegression

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


if __name__ == "__main__":
    points = np.array([[random.randint(70,100) for i in range(3)] for j in range(500)])
    #plot_pareto_frontier(points)
    #scatter_binary(points)
    approximate_upper_bound(points)
