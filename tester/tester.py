__author__ = 'adam'

from registers import *

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
    # <test execution>
    print current

    # </test execution>
    mgr.next()
    if mgr.isOverflowOccured():
        break

