from anytree import Node, RenderTree
id=0

class Tree(object):
    def __init__(self,alg1,alg2,parent,_id):
        self.id=_id
        self.left = None
        self.right = None
        self.printable=None
        self.p1=0.5
        self.p2=0.5
        self.alg1=alg1
        self.alg2=alg2
        self.lastInstanceIndex = -1
        self.parent = parent
        self.visited = False
        self.jp_vs_run = []              #joint probability
        self.p1_vs_run = []
        self.p2_vs_run = []
        self.p_vs_run = []
        self.stat_vs_run = []
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

    def addLeft(self,child):
        global id
        id = id+1
        child.id = id
        self.left=child
        child.parent=self
        child.printable=Node(child.getData(),parent=self.printable)
        print "added left child node"+str(child.id)+" to node"+str(self.id)


    def addRight(self,child):
        global id
        id = id+1
        child.id = id
        self.right=child
        child.parent=self
        child.printable=Node(child.getData(),parent=self.printable)
        print "added right child node"+str(child.id)+" to node"+str(self.id)

    def getData(self):
        return str(self.id)+"__"+self.alg1.name+"("+str(self.p1)+") vs "+self.alg2.name+"("+str(self.p2)+")"

    def printTree(self):
        print("______________________")
        print "imprimiendo arbol"
        if self.printable==None:
            self.printable=Node(self.getData())
        for pre, fill, node in RenderTree(self.printable):
            print("%s%s" % (pre, node.name))
        print("______________________")

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

    def getMsg(self):
        msg = []
        node=self

        if self.p1>self.p2:
            msg.insert(0,0)
        else:
            msg.insert(0,1)

        while(node.parent!=None):
            if node.parent.left == node:
                msg.insert(0,0)
            if node.parent.right == node:
                msg.insert(0,1)
            node=node.parent

        return msg

    def save_jp(self):
        (self.jp_vs_run).append(self.jointProbability())

    def save_p(self):
        (self.p1_vs_run).append(self.p1)
        (self.p2_vs_run).append(self.p2)
    def save_test(self,test,probability):
        self.stat_vs_run.append(tuple((test,probability)))

    def get_jp_vs_run(self):
        return self.jp_vs_run

    def get_p_vs_run(self):
        return (self.p1_vs_run,self.p2_vs_run)
