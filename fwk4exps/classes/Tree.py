from anytree import Node, RenderTree
from decimal import *
#encoding=utf8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


id=0

class Tree(object):
    def __init__(self,alg1,alg2,parent,_id,ins_ord,rangebase):
        self.id=_id
        self.left = None
        self.right = None
        self.pvalue = 0.5 #borrar
        self.printable=None 
        self.p1=0
        self.p2=0
        self.alg1=alg1
        self.alg2=alg2
        self.lastInstanceIndex1 = -1
        self.lastInstanceIndex2 = -1
        self.ins_ord = ins_ord
        self.simulationVisitCount =0
        self.rangebase = rangebase
        #self.times_ran=0
        self.parent = parent
        self.visited = False
        self.m1 =0.5
        self.m2 =0.5
        self.isLeafLeaf = False

    
    def setMaxp(self):
        if self.p1>self.p2:
            self.p1=1
            self.p2=0
        else:
            self.p2=1
            self.p1=0

    def setPvalueZero(self):
        self.pvalue=0

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
        #print "added left child node"+str(child.id)+" to node"+str(self.id)


    def addRight(self,child):
        global id
        id = id+1
        child.id = id
        self.right=child
        child.parent=self
        child.printable=Node(child.getData(),parent=self.printable)
        #print "added right child node"+str(child.id)+" to node"+str(self.id)

    #def refresh(self):
    #    self.simulationVisitCount = 0

    def addSimulationVisit(self):
        self.simulationVisitCount = self.simulationVisitCount + 1
        
    def getData(self):
        #print str(self.id)+"__"+self.alg1.name+"("+str(self.p1)+") vs "+self.alg2.name+"("+str(self.p2)+")"
        p1 = str(Decimal(self.p1).quantize(Decimal('1.000')))
        p2 = str(Decimal(self.p2).quantize(Decimal('1.000')))
        sim = str(self.simulationVisitCount)
        cond = str(self.conditionalProb())
        m1 = str(self.m1)
        m2 = str(self.m2)
        #jp1 = str(Decimal(self.jointProbability()).quantize(Decimal('1.000')))
        #jp2 = str(Decimal(self.jointProbability2()).quantize(Decimal('1.000')))
        #return str(self.id)+"__"+self.alg1.name+"("+str(self.p1)+") vs "+self.alg2.name+"("+str(self.p2)+")"
        return str(self.id)+"__"+self.alg1.name+"("+p1+","+m1+") vs "+self.alg2.name+"("+p2+","+m2+")"+"|simulations: "+sim+"| |conditionalProb = "+ cond +"| instancias corridas:alg1"+str(self.lastInstanceIndex1)+",alg2:"+str(self.lastInstanceIndex2)#+"|("+jp1+")_("+jp2+")"
        #return str(self.id)+"__"+self.alg1.name+" vs "+self.alg2.name+"|simulations: "+sim#+"|("+jp1+")_("+jp2+")"

    def printPreorder(self):
        if self!=None:
            if(self.parent!=None):
                self.printable=Node((self.getData()).encode("utf-8"),parent=self.parent.printable)
            else:
                self.printable=Node((self.getData()).encode("utf-8"))
            # Then recur on left child
            if self.left!=None:
                self.left.printPreorder()

            # Finally recur on right child
            if self.right!=None:
                self.right.printPreorder()

    def refreshSimulations(self):
        if self!=None:
            self.simulationVisitCount = 0
            self.p1 = 0
            self.p2 = 0
            # Then recur on left child
            if self.left!=None:
                self.left.refreshSimulations()

            # Finally recur on right child
            if self.right!=None:
                self.right.refreshSimulations()

    def printTree(self):
        #print("______________________")
        #print "imprimiendo arbol"

        #actualizar
        #if self.printable==None:
        self.printPreorder()
        #self.printable=Node(self.getData())
        #print(RenderTree(self.printable))
        for pre, _, node in RenderTree(self.printable):
            print("%s%s" % (pre, node.name))
        #print("______________________")

    def conditionalProb(self):
        '''
        if self.parent is None:
            return 1
        else:
            if self.simulationVisitCount==0 or self.parent.simulationVisitCount == 0:
                return 0
            #print "self.simulationVisitCount: " + str(self.simulationVisitCount)
            #print "self.parent.simulationVisitCount: " + str(self.parent.simulationVisitCount)
            return self.simulationVisitCount / (self.parent.simulationVisitCount * 1.0)
        '''
        if self.simulationVisitCount ==0:
            return 0
        return self.bestp1p2() / (self.simulationVisitCount *1.0)
    def isLeaf(self):
        return self.left==None and self.right==None

    def isNotLeaf(self):
        return self.left == None or self.right == None
 
    def bestp1p2(self):
        if self.p1>self.p2:
            return self.p1
        else:
            return self.p2

    def worstp1p2(self):
        if self.p1<self.p2:
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

    def executeAlgorithm1(self, i,algid, instance, PI, return_dict):
        return_dict[i,algid] = self.alg1.run(instance, PI)

    
    def executeAlgorithm2(self, i,algid, instance, PI, return_dict):
        return_dict[i,algid] = self.alg2.run(instance, PI)
    '''
    def selectInstance(self): 
        self.lastInstanceIndex=self.lastInstanceIndex + 1
        index=self.lastInstanceIndex
        i=self.ins_ord[index]
        return i
    '''
    def selectInstance1(self): 
        self.lastInstanceIndex1=self.lastInstanceIndex1 + 1
        index=self.lastInstanceIndex1
        i=self.ins_ord[index]
        #print ("index"+str(index))
        #print ("i:"+str(i))
        #print ("largo de arreglo instancias"+str(len(self.ins_ord)))
        if i<= len(self.ins_ord)+self.rangebase:
            return i
        else:
            return None

    def selectInstance2(self): 
        self.lastInstanceIndex2=self.lastInstanceIndex2 + 1
        index=self.lastInstanceIndex2
        i=self.ins_ord[index]
        #print ("index"+str(index))
        #print ("i:"+str(i))
        #print ("largo de arreglo instancias"+str(len(self.ins_ord)))
        if i <= len(self.ins_ord)+self.rangebase:
            return i
        else:
            return None
    
    def isTerminated(self):
        return self.lastInstanceIndex1 == len(self.ins_ord)-1 and self.lastInstanceIndex2 == len(self.ins_ord)-1

    def savemean1(self,m1):
        self.m1=m1

    def savemean2(self,m2):
        self.m2=m2