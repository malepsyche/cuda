import numpy as np
import matplotlib.pyplot as plt
import glob

# Heatmaps
files = sorted(glob.glob("frame_*.txt"))

for f in files:
    data = np.loadtxt(f)
    plt.imshow(data, cmap='viridis')
    plt.colorbar()
    plt.title(f)
    plt.savefig(f.replace(".txt",".png"))
    plt.clf()

# Bandwidth scaling plot
import pandas as pd
df = pd.read_csv("results.csv", names=["L","N","time","bw"])

plt.plot(df["N"], df["bw"], marker='o')
plt.xlabel("Grid Size")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Scaling Performance")
plt.savefig("scaling.png")
plt.show()