__author__ = 'adam'


class CyclicRegister:
    def __init__(self, array):
        self.array = array
        self.count = len(array)
        self.current = 0
        self.overflowOccured = False

    def get(self):
        return self.array[self.current]

    def next(self):
        self.current += 1
        if self.current == self.count:
            self.current = 0
            self.overflowOccured = True
            return True
        return False

    def isOverflowOccured(self):
        return self.overflowOccured


class OverflowManager:
    def __init__(self, registers):
        self.registers = registers

    def next(self):
        for register in self.registers:
            overflowOccured = register.next()
            if not overflowOccured:
                break

    def isOverflowOccured(self):
        return self.registers[-1].isOverflowOccured()

    def getArray(self):
        return [register.get() for register in self.registers]


criteria = {
    "inputSize": [200, 2000, 20000, 200000, 2000000],
    "numBThreads": [1, 2, 4, 8, 16],
    "numBlocks": [1, 10, 100, 300, 400],
    "device": [0, 1]
}

mgr = OverflowManager([
    CyclicRegister(criteria["inputSize"]),
    CyclicRegister(criteria["numBThreads"]),
    CyclicRegister(criteria["numBlocks"]),
    CyclicRegister(criteria["device"])
])

while True:
    current = mgr.getArray()
    print current
    mgr.next()
    if mgr.isOverflowOccured():
        break

