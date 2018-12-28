import os
import sys
import numpy as np
import random as rd
import sympy as sym
from FunctionPlotter import data_plotter
from FunctionPlotter import stats_plotter
from SimulatedAnnealing import SA_model

# First, we solve the Maximum point of continuous function with simulated annealing method
upper_bound=10
lower_bound=-10
mode='Function_continuous'
stats='Fucntion_stats'
x=sym.symbols('x')
function=0.5*x-1*sym.sin(x)+2*sym.cos(2*x)+3*sym.sin(3*x)-4*sym.sin(4*x)+5*sym.sin(5*x)
max_tolerance=50
init_temperature=100
inner_iteration=20
coeff_cooling=0.95
num_in_cluster=10
epoch=20
best_y_list=[]
for curr_epoch in range(epoch):
    x_list,y_list=SA_model(mode,function,upper_bound,lower_bound,max_tolerance,init_temperature,inner_iteration,coeff_cooling,num_in_cluster)
    # data_plotter(mode,function,upper_bound,lower_bound,x_list,y_list)
    best_y_value=y_list[-1]
    best_y_list.append(best_y_value)
stats_plotter(stats,best_y_list)

# Second, we solve the Travelling Salesmen Problem with simulated annealing method
upper_bound=200
lower_bound=0
mode='TSP_discrete'
stats='TSP_stats'
list_length=20
max_tolerance=100
init_temperature=100
inner_iteration=100
coeff_cooling=0.99
epoch=20
node_list=[]
best_distance_list=[]
for conut in range(list_length):
    rd_x=rd.randint(lower_bound,upper_bound)
    rd_y=rd.randint(lower_bound,upper_bound)
    node_list.append(np.array([rd_x,rd_y]))
node_list=np.array(node_list)

for curr_epoch in range(epoch):
    solution_list,distance_list=SA_model(mode,node_list,list_length,max_tolerance,init_temperature,inner_iteration,coeff_cooling)
    # data_plotter(mode,list_length,solution_list,distance_list,lower_bound,upper_bound)
    best_distance=distance_list[-1]
    best_distance_list.append(best_distance)

stats_plotter(stats,best_distance_list)

