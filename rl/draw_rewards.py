#!/usr/bin/env python3
""" plot convergency curves from log files """
import sys
import re
import datetime
import matplotlib.pyplot as plt

print('parameters : %s' % sys.argv)
for i in range(1, len(sys.argv)):
    print(' * add file : ', sys.argv[i])

UPDATE_FREQUENCY = 60*10   # 10 minute

plt.ion()
while True:
    print('last update at {}'.format(datetime.datetime.now()))
    for i in range(1, len(sys.argv)):
        x, y = [], []
        for line in open(sys.argv[i], 'r').readlines():
            m = re.search('^ep (\\d+):.*averageReward: ([0-9\\.-]+)', line)
            if m is None:
                continue
            x.append(int(m.group(1)))
            y.append(float(m.group(2)))
        plt.plot(x, y, label=sys.argv[i][:-4])

    plt.legend()
    #plt.show(block=0)               # show for one time drawing
    plt.pause(UPDATE_FREQUENCY)     # pause for interactive updating
    plt.clf()
