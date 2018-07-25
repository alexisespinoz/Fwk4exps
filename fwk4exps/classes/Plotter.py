import matplotlib.pyplot as plt #graficos

class Plotter():

    def __init__(self):
        self.jp_vs_run=[]
        self.pvalue_vs_run=[]
        self.p_vs_run=[]

    def save_jp(self,node):
        (self.jp_vs_run(node.id)).append(node.jointProbability())

    def save_p(self,node):
        (self.p_vs_run[node.id]).append(node.p1)

    def save_pvalue(self,node):
        self.pvalue_vs_run(node.id).append(node.pvalue)

    def plot_p_vs_run(self):
        a=p_vs_run[:,0]
        b[:]=[1-x for x in a]
        plt.figure()
        plt.plot(a)
        plt.plot(b)
        plt.title("p1 and p2 vs runs (root)")

    def plot_pvalue_vs_run(self):
        a=pvalue_vs_run[:,0]
        plt.figure()
        plt.plot(a)
        plt.title("pvalue vs runs (root)")

    def plot_jp_vs_run(self):
        a=jp_vs_run(0)
        plt.figure()
        plt.plot(a)
        plt.title("joint Probability vs runs (root)")

    def plot_all(self):
        self.plot_p_vs_run()
        self.plot_pvalue_vs_run()
        self.plot_jp_vs_run()
        plt.show()
