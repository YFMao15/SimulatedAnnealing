import os
import sys
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.animation as amn
import numpy as np

# Statistical plotting
def stats_plotter(mode,*args):
    if mode=='TSP_stats':
        plt.ion()
        best_distance_list=args[0]
        list_length=len(best_distance_list)
        np_list=np.array(best_distance_list)
        list_mean=np_list.mean()
        list_variance=np_list.var()
        plt.scatter(range(list_length),best_distance_list,c='red',s=150)
        plt.plot(range(list_length),list_mean*np.ones(np_list.shape),color='green',linewidth=2)
        plt.xlim((0,list_length+1))
        plt.ylim(min(best_distance_list)-20,max(best_distance_list)+20)
        plt.xlabel('Epoches')
        plt.ylabel('Best Distance')
        plt.title('TSP Stats Map')
        plt.legend(['Mean: %.3f, Variance: %.3f' % (list_mean,list_variance)])
        plt.pause(10)
    
    elif mode=='Function_stats':
        plt.ion()
        best_y_list=args[0]
        list_length=len(best_y_list)
        np_list=np.array(best_y_list)
        list_mean=np_list.mean()
        list_variance=np_list.var()
        plt.scatter(range(list_length),best_y_list,c='red',s=150)
        plt.plot(range(list_length),list_mean*np.ones(np_list.shape),color='green',linewidth=2)
        plt.xlim((0,list_length+1))
        plt.ylim(min(best_y_list)*9/10,max(best_y_list)*11/10)
        plt.xlabel('Epoches')
        plt.ylabel('Best Distance')
        plt.title('TSP Stats Map')
        plt.legend(['Mean: %.3f, Variance: %.3f' % (list_mean,list_variance)])
        plt.pause(10)

# Situational plotting
def data_plotter(mode,*args):
    if mode=='TSP_discrete':
        # list_length is the number of the destinations
        # solution_list is all updating logs of travelling paths
        # distance_list is all updating logs of corresponding path distances in solution_list
        # lower bound and upper bound limits the range of the coordinates
        list_length=args[0]
        solution_list=args[1]
        distance_list=args[2]
        lower_bound=args[3]
        upper_bound=args[4]
        
        
        # solution_length is the number of logs
        solution_length=len(solution_list)
        plt.ion()
        for count in range(solution_length):
            plt.clf()
            plt.subplot(121)
            temp_solution=solution_list[count]
            solution_x=np.append(temp_solution[:,0],temp_solution[0,0])
            solution_y=np.append(temp_solution[:,1],temp_solution[0,1])
            plt.xlim((lower_bound,upper_bound))
            plt.ylim((lower_bound,upper_bound))
            plt.title('Situation Map')
            plt.xlabel('X-coordinate',fontsize='large')
            plt.ylabel('Y-coordinate',fontsize='large')
            plt.scatter(solution_x,solution_y,c='red',s=150)
            plt.plot(solution_x,solution_y,color='blue',linewidth=2)
            plt.subplot(122)
            temp_list=distance_list[0:count]
            plt.xlim((0,solution_length+1))
            plt.ylim(min(distance_list)-50,max(distance_list)+50)
            plt.title('Convergence Map')
            plt.xlabel('Iteration Times',fontsize='large')
            plt.ylabel('Distance',fontsize='large')
            plt.scatter(range(1,count+1),temp_list,c='red',s=150)
            plt.plot(range(1,count+1),temp_list,color='red',linewidth=2)
            plt.pause(0.5) 
        plt.pause(10)                              

    elif mode=='Function_continuous':
        function=args[0]
        upper_bound=args[1]
        lower_bound=args[2]
        x_list=args[3]
        y_list=args[4]
        x=sym.symbols('x')
        range_x=np.arange(lower_bound,upper_bound,0.02)
        range_y=[]
        for temp_x in range_x:
            temp_y=function.subs({x:temp_x})
            range_y.append(temp_y)
        range_y=np.array(range_y).astype('float32')
        list_length=len(y_list)

        for count in range(list_length):
            plt.clf()
            plt.subplot(121)
            plt.xlim((lower_bound,upper_bound))
            plt.ylim((min(range_y)*9/10,max(range_y)*11/10))
            plt.title('Function Map')
            plt.xlabel('X-coordinate',fontsize='large')
            plt.ylabel('Y-coordinate',fontsize='large')
            plt.plot(range_x,range_y,color='blue',linewidth=2)
            axis_y=np.array(np.arange(min(range_y),max(range_y),0.1))
            axis_x=np.ones(axis_y.shape)*x_list[count]
            plt.plot(axis_x,axis_y,color='red',linewidth=2)
            plt.subplot(122)
            temp_list=y_list[0:count]
            plt.xlim((0,list_length+1))
            plt.ylim(min(y_list)*9/10,max(y_list)*11/10)
            plt.title('Convergence Map')
            plt.xlabel('Iteration Times',fontsize='large')
            plt.ylabel('Distance',fontsize='large')
            plt.scatter(range(1,count+1),temp_list,c='red',s=150)
            plt.plot(range(1,count+1),temp_list,color='red',linewidth=2)
            plt.pause(0.5) 
        plt.pause(10)



    

        
    





