# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:25:36 2022

@author: syoni
"""

from queue import PriorityQueue


def UCS(graph, start, goal, weights=None):
    queue = PriorityQueue()
    queue.put((0, start, [start]))
    explored = []

    while True:
        if queue.empty():
            raise Exception("graph is empty")

        totalweight, currentnode, path = queue.get()
        explored.append(currentnode)

        if currentnode == goal:
            return "UCS: " + " " .join(path)

        for nextnode in graph[currentnode]:
            if nextnode not in explored:
                weight = weights.get((currentnode, nextnode), 10e100) if weights else 1
                newweight = totalweight + weight
                newpath = path + [nextnode]
                queue.put((newweight, nextnode, newpath))


search_graph = {
    'Sports Complex':['Siwaka'],
    'Siwaka':['Ph.1A','Ph.1B'],
    'Ph.1A':['Mada','Ph.1B'],
    'Ph.1B':['Phase 2','STC'],
    'Phase 2':['J1','Phase 3','STC'],
    'STC':['Parking Lot'],
    'Mada':['Parking Lot'],
    'J1':['Mada'],
    'Parking Lot':[],
    'Phase 3':['Parking Lot'],

}

graph_weights = {
    ('Sports Complex','Siwaka'):450,
    ('Siwaka','Ph.1A'):10,
    ('Ph.1A','Mada'):850,
    ('Siwaka','Ph.1B'):230,
    ('Ph.1A','Ph.1B'):100,
    ('Ph.1B','Phase 2'):112,
    ('Ph.1B','STC'):50,
    ('Phase 2','J1'):600,
    ('Phase 2','Phase 3'):500,
    ('Phase 2','STC'):50,
    ('STC','Parking Lot'):250,
    ('J1','Mada'):200,
    ('Mada','Parking Lot'):700,
    ('Phase 3','Parking Lot'):350,
}

if __name__ == '__main__':
    print(UCS(search_graph, 'Sports Complex', 'Parking Lot', graph_weights))
                
