import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300

losses = []

with open("logs.txt", "r") as fp:
    for line in fp.readlines():
        if "it" in line and "]" in line:
            try:
                loss = float(line.strip().split("]")[-1])
                losses.append(loss)
            except:
                continue

avg = []
for i in range(len(losses) - 32):
    avg.append(sum(losses[i:i+32]) / 32)

plt.plot(avg)
plt.title("GVP-Hgraph Trial 1")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("trial_1.png")
