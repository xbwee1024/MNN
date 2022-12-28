""" python demo usage about MNN API """
from __future__ import print_function
# import numpy as np
# import cv2
import sys
from mnn import Interpreter, Tensor, ScheduleConfig


def main(ifname):
    interpreter = Interpreter.createFromFile(ifname)
    interpreter.setCacheFile('.tmpcache')
    interpreter.setSessionMode(Interpreter.Session_Debug)
    config = ScheduleConfig()
    session = interpreter.createSession(config)
    inputs = interpreter.getSessionInputAll(session)
    outputs = interpreter.getSessionOutputAll(session)
    print("input count: ", inputs.size())
    print("output count: ", outputs.size())
    input = interpreter.getSessionInput(session, None)
    output = interpreter.getSessionOutput(session, None)
    print("input: %dx%dx%dx%d" % (input.batch(),
          input.channel(), input.height(), input.width()))
    print("output: %dx%dx%dx%d" % (output.batch(),
          output.channel(), output.height(), output.width()))


if __name__ == '__main__':
    ifname = sys.argv[1]
    main(ifname)
