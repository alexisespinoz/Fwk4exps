from anytree import Node, RenderTree
class Tree(object):
    def __init__(self,alg1,alg2,parent):
        self.left = None
        self.right = None
        self.p1=None
        self.p2=None
        self.alg1=alg1
        self.alg2=alg2
        self.data = self.alg1.name+"("+str(self.p1)+") vs "+self.alg2.name+"("+str(self.p2)+")"
        self.lastInstanceIndex = -1
        self.parent = parent
        self.visited = False
        #self.jp = 1              #joint probability

    def children():
        vec=[]
        vec.append(self.left)
        vec.append(self.right)
        return vec
    def printTree(self):
        #print("lalalala")

        root = Node(self.data)
        n1 = Node(self.right.data, parent=root)
        n2 = Node(self.left.data, parent=root)
        #recorrer arbol de mejor manera :)
        for pre, fill, node in RenderTree(root):
            print("%s%s" % (pre, node.name))

    def jointProbability():
        if self.parent==None:
            return 1
        if self.parent.left==self:
            return self.parent.jointProbability()*self.parent.p1
        if self.parent.rigth==self:
            return self.parent.jointProbability()*self.parent.p2

    #def leafs():
