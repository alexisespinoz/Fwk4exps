#codigo inspirado en este link: http://anytree.readthedocs.io/en/2.4.3/index.html
# tambien podria ser usado: http://treelib.readthedocs.io/en/latest/examples.html
from anytree import Node, RenderTree

root = Node("a1 vs a2")
n1 = Node("a2 vs a3", parent=root)
n2 = Node("a3 vs a2", parent=root)


#print(udo)
#Node('/Udo')
#print(joe)
#Node('/Udo/Dan/Joe')

for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))


#print(dan.children)
#(Node('/Udo/Dan/Jet'), Node('/Udo/Dan/Jan'), Node('/Udo/Dan/Joe'))
