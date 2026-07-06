import itertools
import numpy as np
import matplotlib.pyplot as plt

# Three musicians: A, B, C
# Each musician has three possible timing states:
# E = early, O = on beat, L = late
timings = ["E", "O", "L"]

# All possible future states of the ensemble
states = ["".join(s) for s in itertools.product(timings, repeat=3)]

# Whole system distribution:
# In a synchronized ensemble, all musicians share the same timing.
# Therefore, only EEE, OOO, and LLL are possible.
p_whole = np.zeros(len(states))

synchronized_states = ["EEE", "OOO", "LLL"]

for sync_state in synchronized_states:
    index = states.index(sync_state)
    p_whole[index] = 1 / 3

# Partitioned distribution:
# If we cut the ensemble into independent musicians,
# each musician has probability 1/3 of being E, O, or L.
# Then all 27 combinations become equally possible.
q_partitioned = np.ones(len(states)) / len(states)

# KL divergence D_KL(p || q)
kl = 0
for p, q in zip(p_whole, q_partitioned):
    if p > 0:
        kl += p * np.log2(p / q)

print(f"Integrated-information-like value: {kl:.3f} bits")

# Plot
x = np.arange(len(states))
width = 0.4

plt.figure(figsize=(14, 6))
plt.bar(x - width / 2, p_whole, width, label="Whole synchronized ensemble")
plt.bar(x + width / 2, q_partitioned, width, label="Partitioned independent musicians")

plt.xticks(x, states, rotation=90)
plt.xlabel("Future timing state of musicians A, B, C")
plt.ylabel("Probability")
plt.title("IIT Example: Synchronization in a Music Ensemble")
plt.legend()

plt.text(
    0.5,
    0.28,
    f"KL divergence = {kl:.3f} bits",
    fontsize=12
)

plt.tight_layout()
plt.savefig("music_sync_plot.png", dpi=300)
plt.show()
