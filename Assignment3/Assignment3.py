#code inspired from http://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php
#worked with Ryan & Conner on this homework assignment
import sys
import time

def fileInput():
    file_data = open("Assignment3.txt")
    content = file_data.readlines()
    diction = {}
    testGraph = Graph()
    
    for lines in content:
        if(lines.find('[') != -1):
            node_data = lines.strip('[').strip('\n').strip('\r').strip(']').split(',')
            if (testGraph.vert_dict.has_key(node_data[0]) == False):
                testGraph.add_vertex(node_data[0])
            if (testGraph.vert_dict.has_key(node_data[1]) == False):
                testGraph.add_vertex(node_data[1])
            testGraph.add_edge(node_data[0], node_data[1], int(node_data[2]))
        else:
            if(lines not in ['\n', '\r\n']):
                h_data = lines.strip('\n').strip('\r').strip(' ').split('=')
                diction[h_data[0]] = int(h_data[1])

    # return (testGraph.nodes, diction)
    return (diction,testGraph)


class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.distance = sys.maxint    
        self.visited = False  
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return

import heapq

def dijkstra(aGraph, start, target):
    print '''Dijkstra's'''
    start.set_distance(0)

    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while target.visited == False:
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        for next in current.adjacent:
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                print 'updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
            else:
                print 'not updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())

        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)

    all_vistit = []
    for key, value in aGraph.vert_dict.items():
        if(value.visited == True):
            all_vistit.append(value.get_id())
    print all_vistit


def aStar(aGraph, start, target, heur):
    start.set_distance(0)

    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while target.visited == False:
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        for next in current.adjacent:
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next) + heur[current.get_id()]
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                print 'updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
            else:
                print 'not updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())

        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)

    all_vistit = []
    for key, value in aGraph.vert_dict.items():
        if(value.visited == True):
            all_vistit.append(value.get_id())
    print all_vistit




if __name__ == '__main__':
    (h,g) = fileInput()
    (k,i) = fileInput()
    


    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print '( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w))
    time_start = time.time()
    dijkstra(g, g.get_vertex('S'), g.get_vertex('F'))
    time_end = time.time()
    target = g.get_vertex('F')
    path = [target.get_id()]
    shortest(target, path)
    print 'Dijkstra'
    print 'Shortest path : %s' %(path[::-1])
    print "Time:", (time_end - time_start), "s"

    time_start = time.time()
    aStar(i, i.get_vertex('S'), i.get_vertex('F'), k)
    time_end = time.time()
    target = i.get_vertex('F')
    path = [target.get_id()]
    shortest(target, path)
    print 'aStar'
    print 'Shortest path : %s' %(path[::-1])
    print "Time:", (time_end - time_start), "s"
