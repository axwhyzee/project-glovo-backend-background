class WeightedQuickUnionPathCompression():
    '''
    Weighted quick union with path compression, 1 pass variant
    '''
    def __init__(self, n: int):
        self.__ids = [i for i in range(n)]
        self.__size = [1 for _ in range(n)]

    def find(self, p):
        while p != self.__ids[p]:
             # attach node p to grandid
            self.__ids[p] = self.__ids[self.__ids[p]]
            p = self.__ids[p]
        
        return p
    
    def union(self, p, q):
        p_id = self.find(p)
        q_id = self.find(q)
        
        if self.__size[p_id] < self.__size[q_id]:
            self.__ids[p] = q_id
            self.__size[q_id] += self.__size[p_id]
        else:
            self.__ids[q] = p_id
            self.__size[p_id] += self.__size[q_id]

    def connected(self, p, q):
        return self.find(p) == self.find(q)
    
    def get_ids(self):
        return self.__ids