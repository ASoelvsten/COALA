import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
import COALA as coala
import mcmc

def create_mockdata_fiji(param=[-4.5,0.0,0.0,0.0,0.0,2.5,2.5,0.7,1.5,0.6,0.3,-0.5,0.0,0.0,0.8,3.0,0.0,0.0,0.0,0.0,0.,0.,0.3,0,0,0], name="./summary_mock_XD.npy"):

    np.random.seed(seed=42)

    # Parameters
    num_large_islands = 2
    num_small_islands = 6
    points_per_large_island = 70
    points_per_small_island = 7
    large_island_radius = 10
    small_island_radius = 0.5

    num_villages = num_large_islands*points_per_large_island+num_small_islands*points_per_small_island
    num_remove = int(0.3*num_villages)

    # Function to create points around an irregular island shape
    def generate_irregular_island(center_x, center_y, radius, num_points):
        f = int(num_points/4)
        theta = np.linspace(0, 0.3 * np.pi, f)+ np.random.normal(0,0.05,f)
        theta = np.append(theta, np.linspace(0.7* np.pi, 0.9 * np.pi, f)+ np.random.normal(0,0.05,f))
        theta = np.append(theta, np.linspace(1.7* np.pi, 2 * np.pi, num_points-2*f)+ np.random.normal(0,0.05,num_points-2*f))
        r = radius * (1 + np.random.normal(0,0.03,num_points))  # Irregular shape
        x_points = center_x + r * np.cos(theta)
        y_points = center_y + r * np.sin(theta)
        return x_points, y_points

    # Define centers for the islands
    island_centers = [
        (10, 10), (-10, -10),
        (20, -20), (-20, 20),
        (15, -10), (15, -10),
        (10, -5),(7, -15)]

    x = []
    y = []
    Island = []
    for i in range(num_large_islands):
        center_x, center_y = island_centers[i]
        x_points, y_points = generate_irregular_island(center_x, center_y, large_island_radius, points_per_large_island)
        x.extend(x_points)
        y.extend(y_points)
        Island.extend([i]*len(x_points))

    for i in range(num_small_islands):
        center_x, center_y = island_centers[num_large_islands + i]
        x_points, y_points = generate_irregular_island(center_x, center_y, small_island_radius, points_per_small_island)
        x.extend(x_points)
        y.extend(y_points)
        Island.extend([num_large_islands+i]*len(x_points))

    indices_remove = np.random.choice(num_villages, num_remove, replace=False)

    x = np.delete(x, indices_remove)
    y = np.delete(y, indices_remove)
    Island = np.delete(Island, indices_remove)
    num_villages = len(x)
    province = np.arange(num_villages)
    district = np.arange(num_villages)

    x = np.array(x)*250000/40+500000
    y = np.array(y)*300000/40+500000

    def survey(prob, num, values = [-2, -1, 0, 1, 2]):
        answer = np.random.choice(values, num, p=prob)
        return answer

    # Define survey variables

    E13  = survey([0.1,0.1,0.35,0.35,0.1],num_villages)
    G121 = [np.random.randint(-2, 2) for _ in range(num_villages)]
    G122 = [np.random.randint(-2, 2) for _ in range(num_villages)]
    G123 = [np.random.randint(-2, 2) for _ in range(num_villages)]
    fairness1 = survey([0.4,0.15,0.15,0.15,0.15],num_villages)
    fairness2 = survey([0.4,0.15,0.15,0.15,0.15],num_villages)
    X02 = survey([0.1,0.1,0.1,0.35,0.35],num_villages)
    X03 = survey([0.1,0.1,0.35,0.35,0.1],num_villages)
    X07 = survey([0.1,0.1,0.1,0.35,0.35],num_villages)
    t = np.zeros(num_villages)
    t[-1] = 2012
    t[-2] = 2003
    adopters = np.zeros((num_villages,10))
    adopters[-1,8] = 1
    adopters[-2,0] = 1
    X01 = np.random.binomial(1, 0.2, num_villages)
    X04 = np.random.binomial(1, 0.8, num_villages)
    X05 = np.random.binomial(1, 0.8, num_villages)
    X08 = np.random.binomial(1, 0.2, num_villages)
    X09 = np.random.exponential(1 / 0.08, num_villages)
    X10 = np.random.binomial(1, 0.2, num_villages)
    informed = X01.copy()
    Q01 = np.random.exponential(1 / 0.2, num_villages)
    Q02 = np.random.exponential(1 / 0.003, num_villages)

    # For simplificity, there are no networks from qoliqoli in this dataset.

    DQ = np.zeros((num_villages, num_villages))
    CC = np.zeros((num_villages, num_villages))

    # Nor are there schools, hospitals or similar.
    # Similarly, we will not define provinces and districts or include connection to the chiefly village.

    X = np.column_stack((X01,X02,X03,X04,X05,X07,X08,X09,X10,E13,Q01,Q02,G121,G122,G123,fairness1,fairness2))

    print(np.shape(X))

    X /= (np.nanmax(X,axis=0) - np.nanmin(X,axis=0))

    def distance(x,y,i):
        dist = np.sqrt((x - x[i])**2+(y - y[i])**2)
        return dist

    # Distance matrix

    for i in range(0,num_villages):
        d = distance(x,y,i)
        dm = np.min(d[np.nonzero(d)])
        d[d>1.5*dm] = 0
        d[d>0] = 1
        d[Island != Island[i]] = 0
        if i == 0:
            D = d.copy()
        else:
            D = np.column_stack((D,d))

    np.fill_diagonal(D,0)

    NGOS = ["NGO1","NGO2","NGO3","NGO4","NGO5","NGO6"]
    prob = [0.35, 0.55, 0.05, 0.05]
    num_ngos = np.random.choice([0, 1, 2, 3], num_villages, p=prob)
    affiliations = [random.sample(NGOS, k) for k in num_ngos]
    for i in range(num_villages):
        row = np.zeros(num_villages)
        if affiliations[i] != []:
            for k in range(num_villages):
                if affiliations[k] != []:
                    common = [ngo for ngo in affiliations[i] if ngo in affiliations[k]]
                    if common != []:
                        row[k] = 1
        if i == 0:
            NGO_network = row.copy()
        else:
            NGO_network = np.column_stack((NGO_network,row))

    data = np.column_stack((x,y,t,Island,province,district,informed,X,D,DQ,NGO_network,CC,adopters))

    np.save(name,data)

    np.save("truth.npy",param)

    pps = param[:-3].copy()

    d, res, ADOPTERS, t, adag, AD, distance_matrix, x, y = coala.distance(1,pps,disttype="heatmap1",profile=True,verbose=True,mode="Fiji",communicate_q=False,communicate_ne=False,communicate_ngo=False)

    adopters = ADOPTERS[1:].T

    time = np.zeros(num_villages)

    for i in range(10):
        for a in range(num_villages):
            if adopters[a,i] == 1 and time[a] == 0:
                time[a] = 2003+i

    data = np.column_stack((x,y,time,Island,province,district,informed,X,D,DQ,NGO_network,CC,adopters))

    np.save(name,data)

    mcmc.likelihood(param,-100*np.ones(len(param)),100*np.ones(len(param)),"Fast_Fiji","heatmap")
    mcmc.likelihood(param,-100*np.ones(len(param)),100*np.ones(len(param)),"Fast_Fiji","Curve")

    mcmc.likelihood(param,-100*np.ones(len(param)),100*np.ones(len(param)),"Fast_Fiji","heatmap")
    mcmc.likelihood(param,-100*np.ones(len(param)),100*np.ones(len(param)),"Fast_Fiji","Curve")
