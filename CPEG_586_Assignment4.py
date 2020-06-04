import sys
from LogisticReg import LogisticReg
from NN import NN
from Utils import Utils

def main():
    utils = Utils()
    X,y = utils.initData() #initialize the data
    lr = LogisticReg()
    lr.createAndTestLogisticReg(X,y)

    nn = NN()
    nn.createAndTestNN(X, y, 3) # last parameter is number of hidden layer neurons


if __name__ == "__main__":
    sys.exit(int(main() or 0))
