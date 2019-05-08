import numpy as np
import networkx as nx
from xlrd import open_workbook
import matplotlib.pyplot as plt
import os
import itertools


for folder_name in ['/Users/paul/Desktop/network_design/',]:
    if os.path.exists(folder_name):
        folder = folder_name
# For Windows, it would also work with double FORWARD slashes
# 'D://Dropbox//Org2.0//Session 3//Analytical Exercise 3//'

# Realized structure
filename = folder + "s3_network_chimera.xlsx"
wb = open_workbook(filename)
s = wb.sheet_by_index(0)

social_data = []
for row in range(s.nrows):
    row_list = []
    for col in range(s.ncols):
        row_list.append(s.cell(row,col).value)
    social_data.append(row_list)

names_list = [x for x in social_data[0][1:]]
name_labels={}
for i in range(len(names_list)):
    name_labels[i] = names_list[i]

social_data = [[x for x in row[1:]] for row in social_data[1:]]
for i in range(len(social_data)):
    for j in range(len(social_data[i])):
        if social_data[i][j] == '':
            social_data[i][j] = 0
social_structure = np.array(social_data)

M = len(social_structure)

# Authority structure
filename = folder + "s3_network_chimera.xlsx"
wb = open_workbook(filename)
s = wb.sheet_by_index(1)

authority_data = []
for row in range(s.nrows):
    row_list = []
    for col in range(s.ncols):
        row_list.append(s.cell(row,col).value)
    authority_data.append(row_list)

authority_data = [[x for x in row[1:]] for row in authority_data[1:]]
for i in range(len(authority_data)):
    for j in range(len(authority_data[i])):
        if authority_data[i][j] == '':
            authority_data[i][j] = 0
authority_structure = np.array(authority_data)

# Common-boss network
common_boss_structure = np.zeros((M,M))
for row in authority_structure:
    subordinates = [i for i, x in enumerate(row) if x == 1]
    for tie in itertools.combinations(subordinates, 2):
        common_boss_structure[tie[0]][tie[1]] = 1
        common_boss_structure[tie[1]][tie[0]] = 1

# Task structure
filename = folder + "s3_network_chimera.xlsx"
wb = open_workbook(filename)
s = wb.sheet_by_index(2)

task_data = []
for row in range(s.nrows):
    row_list = []
    for col in range(s.ncols):
        row_list.append(s.cell(row,col).value)
    task_data.append(row_list)

task_data = [[x for x in row[1:]] for row in task_data[1:]]
for i in range(len(task_data)):
    for j in range(len(task_data[i])):
        if task_data[i][j] == '':
            task_data[i][j] = 0
        elif task_data[i][j] == 1:
            task_data[j][i] = 1
task_structure = np.array(task_data)

#bottom_half = np.zeros(M) # cumulative number of cells available at each row in the matrix's bottom half
#for i in range(M):
#    bottom_half[i] = (i+1)*i/2












# PLOTS

# Authority structure
# Degree
G = nx.from_numpy_matrix(authority_structure, create_using=None)
degrees = nx.degree(G)
degrees_list = [degrees[x]+5 for x in range(len(degrees))]
pos=nx.spring_layout(G, k=0.15)
colors=range(G.number_of_edges(u=None, v=None))
nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color='#A0CBE2',
                                  cmap=plt.cm.coolwarm,
                                  node_size=[(sum(authority_structure[v])*3)**2 for v in range(len(degrees))],
                                  label=None)
nodes.set_edgecolor('white')
edges = nx.draw_networkx_edges(G, pos,
           edge_color='#d3d3d3',
           width=1)
labels = nx.draw_networkx_labels(G, pos, font_size=6, font_color='k', font_family='serif', font_weight='normal', alpha=1.0, labels=name_labels, ax=None)
plt.grid(False)
plt.axis('off')
#plt.savefig(folder + "authority.png", dpi=400)
plt.show()


# Task structure measures

# Degree centrality in task structure
G = nx.from_numpy_matrix(task_structure, create_using=None)
degrees = nx.degree(G)
degrees_list = [degrees[x]+5 for x in range(len(degrees))]
pos=nx.spring_layout(G, k=0.15)
colors=range(G.number_of_edges(u=None, v=None))
nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color='#A0CBE2',
                                  cmap=plt.cm.coolwarm,
                                  node_size=[(sum(task_structure[v])*3)**2 for v in range(len(degrees))],
                                  label=None)
nodes.set_edgecolor('white')
edges = nx.draw_networkx_edges(G, pos,
           edge_color='#d3d3d3',
           width=1)
labels = nx.draw_networkx_labels(G, pos, font_size=6, font_color='k', font_family='serif', font_weight='normal', alpha=1.0, labels=name_labels, ax=None)
plt.grid(False)
plt.axis('off')
#plt.savefig(folder + "task.png", dpi=400)
plt.show()


# Social structure measures

# Degree
G = nx.from_numpy_matrix(social_structure, create_using=None)
degrees = nx.degree(G)

degrees_list = [degrees[x]+5 for x in range(len(degrees))]
pos=nx.spring_layout(G, k=0.15)
colors=range(G.number_of_edges(u=None, v=None))
nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color='#A0CBE2',
                                  cmap=plt.cm.coolwarm,
                                  node_size=[(sum(social_structure[v])*3)**2 for v in range(len(degrees))],
                                  label=None)
nodes.set_edgecolor('white')
edges = nx.draw_networkx_edges(G, pos,
           edge_color='#d3d3d3',
           width=1)
labels = nx.draw_networkx_labels(G, pos, font_size=6, font_color='k', font_family='serif', font_weight='normal', alpha=1.0, labels=name_labels, ax=None)
plt.grid(False)
plt.axis('off')
#plt.savefig(folder + "communication_degree.png", dpi=400)
plt.show()

# Betweenness
G = nx.from_numpy_matrix(social_structure, create_using=None)
betweens = nx.betweenness_centrality(G)

betweens_list = [betweens[x]+5 for x in range(len(betweens))]
pos=nx.spring_layout(G, k=0.15)
colors=range(G.number_of_edges(u=None, v=None))
nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color='#A0CBE2',
                                  cmap=plt.cm.coolwarm,
                                  node_size=[v*10000 for v in betweens.values()],
                                  label=None)
nodes.set_edgecolor('white')
edges = nx.draw_networkx_edges(G, pos,
           edge_color='#d3d3d3',
           width=1)
labels = nx.draw_networkx_labels(G, pos, font_size=6, font_color='k', font_family='serif', font_weight='normal', alpha=1.0, labels=name_labels, ax=None)
plt.grid(False)
plt.axis('off')
#plt.savefig(folder + "communication_betweenness.png", dpi=400)
plt.show()


# Degree centrality in common boss structure
G = nx.from_numpy_matrix(common_boss_structure, create_using=None)
degrees = nx.degree(G)
degrees_list = [degrees[x]+5 for x in range(len(degrees))]
pos=nx.spring_layout(G, k=0.3)
colors=range(G.number_of_edges(u=None, v=None))
nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color='#A0CBE2',
                                  cmap=plt.cm.coolwarm,
                                  node_size=[(sum(common_boss_structure[v])*3)**2 for v in range(len(degrees))],
                                  label=None)
nodes.set_edgecolor('white')
edges = nx.draw_networkx_edges(G, pos,
           edge_color='#d3d3d3',
           width=1)
labels = nx.draw_networkx_labels(G, pos, font_size=6, font_color='k', font_family='serif', font_weight='normal', alpha=1.0, labels=name_labels, ax=None)
plt.grid(False)
plt.axis('off')
#plt.savefig(folder + "common_boss.png", dpi=400)
plt.show()




# Match between task structure and formal+realized social structure
match_matrix = task_structure*2 - (task_structure!= ((authority_structure+social_structure+common_boss_structure)>0))
np.fill_diagonal(match_matrix, 0)

G = nx.from_numpy_matrix(match_matrix, create_using=None)
pos=nx.circular_layout(G)
etrue=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] ==2]
ecomm=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] ==-1]
eom=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] ==1]

nx.draw_networkx_edges(G,pos,
                       edgelist=etrue,
                       edge_color='blue',
                       width=0.5)
nx.draw_networkx_edges(G,pos,
                       edgelist=eom,
                       width=0.5,
                       edge_color='red')
nx.draw_networkx_edges(G,pos,
                       edgelist=ecomm,
                       width=0.5,
                       alpha=0.5,
                       edge_color='red',
                       style='dashed')

node_weights = [len([x for x in y if x%2==0])-1 for y in match_matrix]
node_weights = [x/(len(match_matrix)-1) for x in node_weights]
# to save, change sixe to 500 and label fontsize to 4
nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color=node_weights,
                                  cmap=plt.cm.RdYlBu,
                                  node_size=1000,
                                  label=None)
nodes.set_edgecolor('white')

labels = nx.draw_networkx_labels(G, pos, font_size=6, font_color='k', font_family='serif', font_weight='normal', alpha=1.0, labels=name_labels, ax=None)

plt.grid(False)
plt.axis('off')
#plt.savefig(folder + "task_match.png", dpi=400)
plt.show()


## Performance of the organization
performance = (np.sum(match_matrix==2) + np.sum(match_matrix==0))/(M**2)
print("The organization's performance is " + str(performance))


# Which units have silo issues?
units = [3,	3,	3,	3,	1,	1,	1,	1,	1,	1,	2,	1,	1,	1,	2,	2,	2,	1,	1,	4]

G = nx.from_numpy_matrix(match_matrix, create_using=None)
pos=nx.circular_layout(G)
nx.draw_networkx_edges(G,pos,
                       edgelist=eom,
                       width=0.5,
                       edge_color='red')
nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color=units,
                                  cmap=plt.cm.Set3,
                                  node_size=1000,
                                  label=None)                       
nodes.set_edgecolor('white')

labels = nx.draw_networkx_labels(G, pos, font_size=6, font_color='k', font_family='serif', font_weight='normal', alpha=1.0, labels=name_labels, ax=None)

plt.grid(False)
plt.axis('off')
plt.savefig(folder + "silos.png", dpi=400)
plt.show()

