import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/experiment_v2_results.csv")

# Filter for random type
df_random = df[df["type"] == "random"]

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_random, x="reps", y="p_base", label="Base Model (Leakage)", marker="o")
sns.lineplot(data=df_random, x="reps", y="p_abl", label="Ablated Model", marker="o")
plt.title("Pattern Leakage in 'Random' Prompts (GPT-2 Small)")
plt.ylabel("Probability of Pattern Token")
plt.xlabel("Number of Pattern Repetitions in Context")
plt.yscale("log")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig("results/leakage_plot.png")

# Calculate average reduction
df_random["reduction_ratio"] = df_random["p_base"] / df_random["p_abl"]
print("Average leakage reduction ratio (Base Prob / Ablated Prob):")
print(df_random.groupby("reps")["reduction_ratio"].mean())

# Entropy analysis
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_random, x="reps", y="e_base", label="Base Model Entropy", marker="o")
sns.lineplot(data=df_random, x="reps", y="e_abl", label="Ablated Model Entropy", marker="o")
plt.title("Output Entropy: Base vs Ablated")
plt.ylabel("Entropy (bits?)")
plt.xlabel("Number of Pattern Repetitions")
plt.savefig("results/entropy_plot.png")

print("Sample Entropy and Probs (Random Type):")
print(df_random[["reps", "p_base", "e_base", "p_abl", "e_abl"]].head(10))

print("Average entropy change:")
print((df_random["e_abl"] - df_random["e_base"]).mean())
