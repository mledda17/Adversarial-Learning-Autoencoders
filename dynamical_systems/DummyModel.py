class DummyModel:
    def __init__(self,nonLinearInputChar=False, useExternalDataset=False):
        print("***Warning:dummyModel***")

    def stateMap(self, xk, u):
        print("***Warning:dummyModel***")
        return 0

    def outputMap(self, xk):
        print("***Warning:dummyModel***")
        return 0

    def systemDynamics(self, dim, flag=True, U=-1):
        print("***Warning:dummyModel***")
        return 0, 0

    def loop(self, x_k, duk):
        print("***Warning:dummyModel***")
        return 0, 0

    def setU(self, U, U_val):
        print("***Warning:dummyModel***")

    def prepareDataset(self, sizeT, sizeV):
        print("***Warning:dummyModel***")
        return 0, 0, 0, 0