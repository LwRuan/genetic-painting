import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import re

def plot(path):
    losses = []
    with open(path, "r") as file:
        line = file.readline()
        while line:
            pattern = "^stage: (.*) loss: (.*)$"
            # stage = int(re.match(pattern, line).group(1))
            loss = float(re.match(pattern, line).group(2))
            losses.append(loss)
            line = file.readline()
    plt.figure(dpi=160)
    plt.xlabel("#iterations")
    plt.ylabel("loss")
    plt.plot(losses)
    plt.savefig("seagull-loss.png")
    plt.show()

if __name__ == '__main__':
    plot("../bin/seagull2.txt")
    