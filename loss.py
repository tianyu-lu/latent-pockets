import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300


def read_losses(fname):
    losses = []

    with open(fname, "r") as fp:
        for line in fp.readlines():
            losses.append(float(line.strip()))

    avg = []
    for i in range(len(losses) - 200):
        avg.append(sum(losses[i:i+200]) / 200)

    return avg


losses = read_losses("losses_1.txt")[:14000]
plt.plot(losses, label="Frozen decoder")
losses = read_losses("losses_2.txt")[:14000]
plt.plot(losses, label="Trainable decoder")
losses = read_losses("losses_3.txt")[:14000]
plt.plot(losses, label="No Pretraining")

plt.title("GVP-Hgraph")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.savefig("trial_comp.png")
