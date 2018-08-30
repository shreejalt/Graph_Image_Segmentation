import numpy as np
class disjoint_set_2():
    def __init__(self, num_elements):
        self.dist_elements = num_elements
        self.node_mat = np.empty(shape=(num_elements,3), dtype=int)
        for i in range(num_elements):
            self.node_mat[i,0] = 0 #Rank
            self.node_mat[i,1] = 1 #Size
            self.node_mat[i,2] = i #Parent
            
            
    def size(self, inp):
        return self.node_mat[inp,1]
    
    def total_sets(self):
        return self.dist_elements
    
    def find_set(self, inp):
        temp = int(inp)
        
        while temp != self.node_mat[temp,2]:
            temp = self.node_mat[temp,2]
        
        self.node_mat[inp,2] = temp #Path compression
        return temp
    
    def join_set(self, inp1, inp2):
        if self.node_mat[inp1,0] > self.node_mat[inp2,0]:
            self.node_mat[inp1,1]+=self.node_mat[inp2,1]
            self.node_mat[inp2, 2] = inp1
        else:
            self.node_mat[inp2,1]+=self.node_mat[inp1,1]
            self.node_mat[inp1, 2] = inp2
            
            if self.node_mat[inp1,0] == self.node_mat[inp2,0]:
                self.node_mat[inp1, 0] = self.node_mat[inp2,0] + 1
        self.dist_elements-=1

            
        
        
            