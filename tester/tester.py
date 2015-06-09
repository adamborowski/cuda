__author__ = 'adam'

import os

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
    conf = {
        "inputSize": current[0],
        "numBThreads": current[1],
        "numBlocks": current[2],
        "device": current[3]
    }

    cmd = "Debug/CUDAProj 128 " \
          "{0[numBThreads]} 128 {0[numBlocks]} {0[device]} {0[inputSize]}" \
          " | grep -oPe \"(?<=process time: )(.*)(?= ms)\"".format(conf)
    results = []
    numTests = 10
    avg = 0
    # gather time results and calculate average
    for i in range(1, numTests):
        time = float(os.popen(cmd).read())
        avg += time
        results.append(str(time))
    avg /= numTests

    outputLine = [str(c) for c in current] + results + [str(avg)]

    print "\t".join(outputLine)

    # </test execution>
    mgr.next()
    if mgr.isOverflowOccured():
        break

