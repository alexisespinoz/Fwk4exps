from anytree import Node, RenderTree
class Tree(object):
    def __init__(self,alg1,alg2,parent,_id):
        self.id=_id
        self.left = None
        self.right = None
        self.p1=0.5
        self.p2=0.5
        self.alg1=alg1
        self.alg2=alg2
        self.lastInstanceIndex = -1
        self.parent = parent
        self.visited = False
        self.jp_vs_run = []              #joint probability

    def setMaxp(self):
        if self.p1>self.p2:
            self.p1=1
            self.p2=0
        else:
            self.p2=1
            self.p1=0
    def children():
        vec=[]
        vec.append(self.left)
        vec.append(self.right)
        return vec

    def getData(self):
        return self.alg1.name+"("+str(self.p1)+") vs "+self.alg2.name+"("+str(self.p2)+")"

    def printTree(self):
        #print("lalalala")
        root = Node(self.getData())
        n1 = Node(self.right.getData(), parent=root)
        n2 = Node(self.left.getData(), parent=root)
        #recorrer arbol de mejor manera :)
        for pre, fill, node in RenderTree(root):
            print("%s%s" % (pre, node.name))

    def jointProbability(self):
        if self.parent==None:
            return self.bestp1p2()
        node=self
        jp=1
        if node.p1>node.p2:
            jp=node.p1
        else:
            jp=node.p2
        while (node.parent!=None):
            if node.parent.left==node:
                 jp=jp*node.parent.p1
            if node.parent.right==node:
                 jp=jp*node.parent.p2
            node=node.parent
        return jp

    def isLeaf(self):
        return self.left==None and self.right==None

    def bestp1p2(self):
        if self.p1>self.p2:
            return self.p1
        else:
            return self.p2

    def save_jp(self):
        (self.jp_vs_run).append(self.jointProbability())

    def get_jp_vs_run(self):
        return self.jp_vs_run
