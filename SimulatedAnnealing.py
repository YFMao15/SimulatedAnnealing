import os
import sys
import sympy as sym
import numpy as np
import random as rd

# generate the neighboring cluster for one specific x_value
    # return both x and y clusters
def cluster_search(x_value,function,num_in_cluster,upper_bound,lower_bound):
    x_cluster=[]
    y_cluster=[]
    x=sym.symbols('x')
    for count in range(num_in_cluster):
        if x_value==0:
            temp_x=rd.uniform(lower_bound,upper_bound)
            temp_y=function.subs({x:temp_x})
        else:
            temp_x=rd.uniform(-2,2)*x_value
            temp_y=function.subs({x:temp_x})
            while (temp_x>upper_bound)|(temp_x<lower_bound):
                temp_x=rd.uniform(-2,2)*x_value
                temp_y=function.subs({x:temp_x})
        x_cluster.append(temp_x)
        y_cluster.append(temp_y)
    
    return np.array(x_cluster).astype('float32'),np.array(y_cluster).astype('float32')

# calculate the distance of current node_list of TSP_discrete
def calculate_distance(node_list,list_length):
    distance=0
    for count in range(list_length-1):
        interval=np.sum(np.square((node_list[count]-node_list[count+1])))
        distance+=interval
    distance+=np.sum(np.square((node_list[0]-node_list[list_length-1])))
    distance=np.sqrt(distance)
    return distance

# search the neighbors of current node_list of TSP_discrete
def neighbor_search(node_list,list_length):
    neighborhood_list=[]
    for count in range(list_length):
        pos1=count
        pos2=rd.randint(0,list_length-1)
        while (pos1==pos2):
            pos2=rd.randint(0,list_length-1)
        # only one exchange from node_list
        temp_list=node_list.copy()
        temp_list[pos2]=node_list[pos1]
        temp_list[pos1]=node_list[pos2]
        neighborhood_list.append(temp_list)
    return neighborhood_list

def SA_model(mode,*args):
    if mode=='TSP_discrete':
        # node_list is the numpy array of all destinations of the salesman
        # list_length is the number of the destinations
        # max_tolerance is the maximum number of time tolerating no new findings on better solutions
            # the current solution will be regraded as the global best solution if the max_iteration is reached
        # init_temperature is the initial temperature of SA
        # inner_iteration is the iterating time for each temperature
        # coeff_cooling is the cooling coefficient on temperature
            # every inner_iteration times the temperature cool down by multiplying this coefficient
        node_list=args[0]
        list_length=args[1]
        max_tolerance=args[2]
        init_temperature=args[3]
        inner_iteration=args[4]
        coeff_cooling=args[5]
        
        # initialization
        distance_list=[]
        solution_list=[]      
        tolerance=0 
        iter_count=0
        temperature=init_temperature

        best_distance=calculate_distance(node_list,list_length)
        best_list=node_list.copy()
        # print('The initial solution is %.3f' % best_distance)
        distance_list.append(best_distance)
        solution_list.append(best_list)

        while tolerance<max_tolerance:           
            tolerance+=1
            iter_count+=1
            if iter_count % inner_iteration==0:
                temperature*=coeff_cooling

            neighborhood_list=neighbor_search(node_list,list_length)
            temp_distance_list=[]
            for count in range(list_length):
                temp_distance=calculate_distance(neighborhood_list[count],list_length)
                # update the current solution if a better choice is found
                temp_distance_list.append(temp_distance)

            min_distance=min(temp_distance_list)
            if min_distance<best_distance:
                index=np.argwhere(temp_distance_list==min_distance)                   
                best_list=neighborhood_list[index[0,0]].copy()
                node_list=neighborhood_list[index[0,0]]
                best_distance=min_distance
                distance_list.append(best_distance)
                solution_list.append(best_list)
                tolerance=0
                # print('Iteration %d, the solution updates to %.3f' % (iter_count,best_distance))

            else:
                # all new choices are worse than the current solution
                # accept the choice based on the evaluation of temperature and threshold
                threshold=rd.uniform(0,1)
                prob_list=np.exp(-(np.array(temp_distance_list)-np.ones(list_length)*best_distance)/temperature) 
                if threshold<max(prob_list):
                    index=np.argwhere(prob_list==max(prob_list))
                    node_list=neighborhood_list[index[0,0]]             
        return solution_list,distance_list

    elif mode=='Function_continuous':
        # function is the sympy representation of function with variable x
        # upper_bound is the maximum value of x
        # lower_bound is the minimum value of x
        # max_tolerance is the maximum number of time tolerating no new findings on better solutions
            # the current solution will be regraded as the global best solution if the max_iteration is reached
        # init_temperature is the initial temperature of SA
        # inner_iteration is the iterating time for each temperature
        # coeff_cooling is the cooling coefficient on temperature
            # every inner_iteration times the temperature cool down by multiplying this coefficient
        function=args[0]
        upper_bound=args[1]
        lower_bound=args[2]
        max_tolerance=args[3]
        init_temperature=args[4]
        inner_iteration=args[5]
        coeff_cooling=args[6]
        num_in_cluster=args[7]

        # initialization
        y_list=[]
        x_list=[]      
        tolerance=0 
        iter_count=0
        temperature=init_temperature

        x=sym.symbols('x')
        x_value=0
        best_x_value=x_value
        best_y_value=function.subs({x:x_value})
        print('The initial solution is %.3f' % best_y_value)
        x_list.append(best_x_value)
        y_list.append(best_y_value)

        while tolerance<max_tolerance:           
            tolerance+=1
            iter_count+=1
            if iter_count % inner_iteration==0:
                temperature*=coeff_cooling

            x_cluster,y_cluster=cluster_search(x_value,function,num_in_cluster,upper_bound,lower_bound)    
            max_y_value=max(y_cluster)

            if max_y_value>best_y_value:
                index=np.argwhere(y_cluster==max_y_value)
                x_value=x_cluster[index[0,0]]
                best_y_value=max_y_value                   
                x_list.append(x_cluster[index[0,0]])
                y_list.append(y_cluster[index[0,0]])
                tolerance=0
                print('Iteration %d, the solution updates to %.3f' % (iter_count,best_y_value))

            else:
                # all new choices are worse than the current solution
                # accept the choice based on the evaluation of temperature and threshold
                threshold=rd.uniform(0,1)
                prob_list=np.exp((y_cluster-np.ones(num_in_cluster)*best_y_value)/temperature)
                if threshold<max(prob_list):
                    index=np.argwhere(prob_list==max(prob_list))
                    x_value=x_cluster[index[0,0]]
                else:
                    x_value=x_value

        return np.array(x_list),np.array(y_list).astype('float32')


