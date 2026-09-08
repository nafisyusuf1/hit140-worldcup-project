"""
======================================================================
 HIT140 Group Project - FIFA World Cup 2026
 FULL STATISTICAL ANALYSIS (all four tasks)
======================================================================

This one script does everything the assignment brief asks for, for
each of the four tasks:

  1. DATA PREPARATION AND SAMPLING
     - Load the data from the csv file.
     - Define the population (the full group of players/matches/teams
       we are interested in) and check that the inclusion rules from
       the brief are actually true in the data.
     - Take a sample from that population. If the population itself is
       already small, we simply use all of it (this is called a
       "census" - a sample that covers the whole population).

  2. DESCRIPTIVE STATISTICS
     - The mean (average), standard deviation (how spread out the
       numbers are), and a few other simple summary numbers, for each
       group we are comparing.
     - A histogram (a bar chart showing how often each range of values
       occurs) for each group.
     - A boxplot, which puts both groups on one chart so you can
       compare them at a glance.

  3. CONFIDENCE INTERVAL
     - A 95% confidence interval for the difference between the two
       group means. In plain words: "we are 95% confident that the
       true difference between the two groups, in the whole
       population, falls somewhere in this range."

  4. TWO-SAMPLE T-TEST
     - A statistical test that tells us whether the difference we see
       between the two groups in our sample is likely to reflect a
       real difference in the population, or could just be random
       noise from which players/matches/teams happened to end up in
       our sample.

HOW THIS FILE IS WRITTEN
-------------------------
The same four steps are repeated separately for each of the four
tasks, using plain, sequential code (no custom functions, no loops
over tasks). This makes the file longer, but it means you can read
Task 1 from top to bottom, understand it fully, and then read Task 2
without needing to jump around the file or understand any extra
programming concepts.

Every result the assignment needs (population size, sample size,
descriptive statistics, confidence interval, t-test result) is printed
to the screen when you run this script, and every chart is saved as a
.png image file so you can drop it straight into your report.
"""

# ----------------------------------------------------------------------
# LIBRARIES WE NEED (these are pre-written tools other people built,
# so we do not have to write our own maths and charting code)
# ----------------------------------------------------------------------
import pandas as pd              # for loading and working with tables of data
import numpy as np               # for maths (square roots, etc.)
import matplotlib.pyplot as plt  # for drawing charts
from scipy import stats          # for the t-test and t-distribution values

# A "seed" makes random sampling repeatable. Anyone who runs this
# script with the same seed number will get exactly the same random
# sample we did - this is important so our results can be reproduced.
RANDOM_SEED = 42

# Earlier in this project, a power analysis (a calculation that tells
# you how big a sample you need) showed that a two-sample t-test needs
# roughly 64 observations per group to reliably detect a medium-sized
# difference between two groups, at 95% confidence and 80% power. We
# use that number for the two tasks where the population is large
# enough to sample from (Tasks 1 and 4).
SAMPLE_SIZE_PER_GROUP = 64

# We are using a 95% confidence level, which means alpha (the amount of
# risk we accept of being wrong) is 0.05.
ALPHA = 0.05

# Folders where the input files live, and where we save our outputs
UPLOAD_DIR = "data"
OUTPUT_DIR = "outputs"


# ======================================================================
# TASK 1: DEFENSIVE WORK RATE BY TOURNAMENT STAGE
# Question: do players from teams that reached the knockout stage make
# more defensive actions (tackles + interceptions) per 90 minutes than
# players from teams eliminated in the group stage?
# ======================================================================
print("\n" + "#" * 70)
print("# TASK 1: DEFENSIVE WORK RATE BY TOURNAMENT STAGE")
print("#" * 70)

# ---------- STEP 1: DATA PREPARATION AND SAMPLING ----------

# Load the csv file. comment="#" tells pandas to skip the lines at the
# top of the file that start with "#" - those are just notes about
# where the data came from, not actual data rows.
task1_raw = pd.read_csv(f"{UPLOAD_DIR}/task1_real_data_1.csv", comment="#")

# INCLUSION RULE (from the brief): outfield players only (no
# goalkeepers), with at least 180 minutes played. The minutes filter is
# already applied in the file. Two rows have a missing "position"
# value, so we cannot confirm those two players are outfield players -
# we remove them to satisfy the inclusion rule properly.
task1_population = task1_raw.dropna(subset=["position"])

print("Population size (after applying inclusion rules):", len(task1_population))
print(task1_population["tournament_stage"].value_counts().to_string())

# The two groups we are comparing
task1_group_A_name = "Advanced to Knockout"
task1_group_B_name = "Eliminated in Group Stage"

task1_population_A = task1_population[task1_population["tournament_stage"] == task1_group_A_name]
task1_population_B = task1_population[task1_population["tournament_stage"] == task1_group_B_name]

# Take a random sample of SAMPLE_SIZE_PER_GROUP players from each group.
# random_state=RANDOM_SEED makes this sample reproducible.
task1_sample_A = task1_population_A.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)
task1_sample_B = task1_population_B.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)

print("Sample size per group:", SAMPLE_SIZE_PER_GROUP)

# The column holding the number we actually want to compare
task1_outcome_column = "defensive_actions_per90"
task1_values_A = task1_sample_A[task1_outcome_column]
task1_values_B = task1_sample_B[task1_outcome_column]

# ---------- STEP 2: DESCRIPTIVE STATISTICS ----------

print("\n--- Descriptive statistics:", task1_outcome_column, "---")
print(task1_group_A_name, "-> n =", len(task1_values_A),
      ", mean =", round(task1_values_A.mean(), 3),
      ", std =", round(task1_values_A.std(), 3),
      ", median =", round(task1_values_A.median(), 3))
print(task1_group_B_name, "-> n =", len(task1_values_B),
      ", mean =", round(task1_values_B.mean(), 3),
      ", std =", round(task1_values_B.std(), 3),
      ", median =", round(task1_values_B.median(), 3))

# Histogram: one small chart per group, placed side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].hist(task1_values_A, bins=15, color="steelblue", edgecolor="white")
axes[0].set_title(task1_group_A_name)
axes[0].set_xlabel(task1_outcome_column)
axes[0].set_ylabel("Number of players")
axes[1].hist(task1_values_B, bins=15, color="darkorange", edgecolor="white")
axes[1].set_title(task1_group_B_name)
axes[1].set_xlabel(task1_outcome_column)
fig.suptitle("Task 1: Histogram of " + task1_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task1_histogram.png", dpi=150)
plt.close(fig)

# Boxplot: both groups on the same chart so they are easy to compare
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([task1_values_A, task1_values_B],
           tick_labels=[task1_group_A_name, task1_group_B_name])
ax.set_ylabel(task1_outcome_column)
ax.set_title("Task 1: Boxplot of " + task1_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task1_boxplot.png", dpi=150)
plt.close(fig)
print("Saved task1_histogram.png and task1_boxplot.png")

# ---------- STEP 3: 95% CONFIDENCE INTERVAL FOR THE DIFFERENCE IN MEANS ----------

# We want a range we are 95% confident contains the TRUE difference
# between the two groups' average defensive actions per 90, in the
# whole population (not just our sample).

task1_mean_A = task1_values_A.mean()
task1_mean_B = task1_values_B.mean()
task1_mean_difference = task1_mean_A - task1_mean_B

task1_n_A = len(task1_values_A)
task1_n_B = len(task1_values_B)
task1_var_A = task1_values_A.var()   # variance = standard deviation squared
task1_var_B = task1_values_B.var()

# Standard error of the difference between two means. We use Welch's
# method here, which does NOT assume the two groups have equal spread
# (variance). This is the safer choice, and it still works fine even
# when the variances happen to be equal.
task1_standard_error = np.sqrt((task1_var_A / task1_n_A) + (task1_var_B / task1_n_B))

# Degrees of freedom for Welch's method. This is a standard formula -
# all it does is work out how wide the t-distribution should be, based
# on our two sample sizes and variances.
task1_deg_freedom = ((task1_var_A / task1_n_A + task1_var_B / task1_n_B) ** 2) / (
    ((task1_var_A / task1_n_A) ** 2) / (task1_n_A - 1)
    + ((task1_var_B / task1_n_B) ** 2) / (task1_n_B - 1)
)

# The "critical value" that marks off the middle 95% of the t-distribution
task1_t_critical = stats.t.ppf(1 - ALPHA / 2, task1_deg_freedom)

task1_margin_of_error = task1_t_critical * task1_standard_error
task1_ci_lower = task1_mean_difference - task1_margin_of_error
task1_ci_upper = task1_mean_difference + task1_margin_of_error

print("\n--- 95% Confidence Interval for the difference in means ---")
print("Mean difference (Advanced - Eliminated):", round(task1_mean_difference, 3))
print("95% CI: [", round(task1_ci_lower, 3), ",", round(task1_ci_upper, 3), "]")

# ---------- STEP 4: TWO-SAMPLE T-TEST ----------

# H0 (null hypothesis):        there is no real difference between the
#                               two groups' average defensive actions per 90.
# H1 (alternative hypothesis): there IS a real difference.

task1_t_statistic, task1_p_value = stats.ttest_ind(
    task1_values_A, task1_values_B, equal_var=False  # Welch's t-test
)

print("\n--- Two-sample t-test (Welch's t-test) ---")
print("t-statistic:", round(task1_t_statistic, 3))
print("p-value:", round(task1_p_value, 5))

if task1_p_value < ALPHA:
    print(f"p-value < {ALPHA}, so we REJECT the null hypothesis.")
    print("Conclusion: there IS a statistically significant difference "
          "between the two groups.")
else:
    print(f"p-value >= {ALPHA}, so we FAIL TO REJECT the null hypothesis.")
    print("Conclusion: we do NOT have enough evidence of a real "
          "difference between the two groups.")


# ======================================================================
# TASK 2: DISCIPLINARY BEHAVIOUR ACROSS MATCH STAGES
# Question: do knockout-stage matches have more yellow cards per match
# than group-stage matches?
# ======================================================================
print("\n" + "#" * 70)
print("# TASK 2: DISCIPLINARY BEHAVIOUR ACROSS MATCH STAGES")
print("#" * 70)

# ---------- STEP 1: DATA PREPARATION AND SAMPLING ----------

task2_raw = pd.read_csv(f"{UPLOAD_DIR}/task2_real_data_csvArik.csv", comment="#")

# INCLUSION RULE: all matches played, excluding abandoned/forfeited
# matches. The file already only contains matches that were played.
task2_population = task2_raw

print("Population size:", len(task2_population))
print(task2_population["knockout_stage"].value_counts().to_string())

print("\nIMPORTANT LIMITATION: this file only has 10 matches, not the "
      "full set of matches played at the tournament, and the matches "
      "look hand-picked rather than randomly chosen (4 of the 10 "
      "involve Argentina). There is no larger pool of matches to "
      "sample from here, so we use all 10 rows as our sample (this is "
      "called a 'census'). With only 5 matches per group, this test "
      "has very little statistical power - treat the result below as "
      "exploratory, not conclusive, and say so in the report.")

task2_group_A_name = "Knockout stage"
task2_group_B_name = "Group stage"

# knockout_stage is a column of True/False values in the data
task2_sample_A = task2_population[task2_population["knockout_stage"] == True]
task2_sample_B = task2_population[task2_population["knockout_stage"] == False]

print("Sample size:", task2_group_A_name, "=", len(task2_sample_A),
      ",", task2_group_B_name, "=", len(task2_sample_B))

task2_outcome_column = "total_yellow"
task2_values_A = task2_sample_A[task2_outcome_column]
task2_values_B = task2_sample_B[task2_outcome_column]

# ---------- STEP 2: DESCRIPTIVE STATISTICS ----------

print("\n--- Descriptive statistics:", task2_outcome_column, "---")
print(task2_group_A_name, "-> n =", len(task2_values_A),
      ", mean =", round(task2_values_A.mean(), 3),
      ", std =", round(task2_values_A.std(), 3),
      ", median =", round(task2_values_A.median(), 3))
print(task2_group_B_name, "-> n =", len(task2_values_B),
      ", mean =", round(task2_values_B.mean(), 3),
      ", std =", round(task2_values_B.std(), 3),
      ", median =", round(task2_values_B.median(), 3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].hist(task2_values_A, bins=8, color="steelblue", edgecolor="white")
axes[0].set_title(task2_group_A_name)
axes[0].set_xlabel(task2_outcome_column)
axes[0].set_ylabel("Number of matches")
axes[1].hist(task2_values_B, bins=8, color="darkorange", edgecolor="white")
axes[1].set_title(task2_group_B_name)
axes[1].set_xlabel(task2_outcome_column)
fig.suptitle("Task 2: Histogram of " + task2_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task2_histogram.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([task2_values_A, task2_values_B],
           tick_labels=[task2_group_A_name, task2_group_B_name])
ax.set_ylabel(task2_outcome_column)
ax.set_title("Task 2: Boxplot of " + task2_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task2_boxplot.png", dpi=150)
plt.close(fig)
print("Saved task2_histogram.png and task2_boxplot.png")

# ---------- STEP 3: 95% CONFIDENCE INTERVAL FOR THE DIFFERENCE IN MEANS ----------

task2_mean_A = task2_values_A.mean()
task2_mean_B = task2_values_B.mean()
task2_mean_difference = task2_mean_A - task2_mean_B

task2_n_A = len(task2_values_A)
task2_n_B = len(task2_values_B)
task2_var_A = task2_values_A.var()
task2_var_B = task2_values_B.var()

task2_standard_error = np.sqrt((task2_var_A / task2_n_A) + (task2_var_B / task2_n_B))

task2_deg_freedom = ((task2_var_A / task2_n_A + task2_var_B / task2_n_B) ** 2) / (
    ((task2_var_A / task2_n_A) ** 2) / (task2_n_A - 1)
    + ((task2_var_B / task2_n_B) ** 2) / (task2_n_B - 1)
)

task2_t_critical = stats.t.ppf(1 - ALPHA / 2, task2_deg_freedom)
task2_margin_of_error = task2_t_critical * task2_standard_error
task2_ci_lower = task2_mean_difference - task2_margin_of_error
task2_ci_upper = task2_mean_difference + task2_margin_of_error

print("\n--- 95% Confidence Interval for the difference in means ---")
print("Mean difference (Knockout - Group):", round(task2_mean_difference, 3))
print("95% CI: [", round(task2_ci_lower, 3), ",", round(task2_ci_upper, 3), "]")
print("Notice how wide this interval is compared to Task 1 - that is a "
      "direct result of having only 5 matches per group.")

# ---------- STEP 4: TWO-SAMPLE T-TEST ----------

task2_t_statistic, task2_p_value = stats.ttest_ind(
    task2_values_A, task2_values_B, equal_var=False
)

print("\n--- Two-sample t-test (Welch's t-test) ---")
print("t-statistic:", round(task2_t_statistic, 3))
print("p-value:", round(task2_p_value, 5))

if task2_p_value < ALPHA:
    print(f"p-value < {ALPHA}, so we REJECT the null hypothesis.")
    print("Conclusion: there IS a statistically significant difference "
          "between the two groups.")
else:
    print(f"p-value >= {ALPHA}, so we FAIL TO REJECT the null hypothesis.")
    print("Conclusion: we do NOT have enough evidence of a real "
          "difference between the two groups (this may be because "
          "there genuinely is no difference, OR because our sample of "
          "10 matches is too small to detect one - we cannot tell "
          "these two explanations apart from this test alone).")


# ======================================================================
# TASK 3: POSSESSION AND SHOOTING DISCIPLINE
# Question: do teams with above-median possession have a higher
# shot-on-target percentage than teams with below-median possession?
# ======================================================================
print("\n" + "#" * 70)
print("# TASK 3: POSSESSION AND SHOOTING DISCIPLINE")
print("#" * 70)

# ---------- STEP 1: DATA PREPARATION AND SAMPLING ----------

task3_raw = pd.read_csv(f"{UPLOAD_DIR}/task3_final_wrangled_csvRatul.csv", comment="#")

# INCLUSION RULE: all teams, excluding any team with incomplete
# possession or shooting data. Drop any row missing those values.
task3_population = task3_raw.dropna(subset=["Poss", "SoTpct"])

print("Population size:", len(task3_population))
print(task3_population["group"].value_counts().to_string())

print("\nNOTE: the population here is all 48 teams at the tournament. "
      "That is already smaller than the ~64-per-group figure our power "
      "analysis called for, and there is no larger population of "
      "'teams at this World Cup' to sample from. So we use the full "
      "census (all 48 teams) rather than throwing away data by "
      "sampling down further.")

task3_group_A_name = "Above median"
task3_group_B_name = "Below median"

task3_sample_A = task3_population[task3_population["group"] == task3_group_A_name]
task3_sample_B = task3_population[task3_population["group"] == task3_group_B_name]

print("Sample size:", task3_group_A_name, "=", len(task3_sample_A),
      ",", task3_group_B_name, "=", len(task3_sample_B))

task3_outcome_column = "SoTpct"
task3_values_A = task3_sample_A[task3_outcome_column]
task3_values_B = task3_sample_B[task3_outcome_column]

# ---------- STEP 2: DESCRIPTIVE STATISTICS ----------

print("\n--- Descriptive statistics:", task3_outcome_column, "---")
print(task3_group_A_name, "-> n =", len(task3_values_A),
      ", mean =", round(task3_values_A.mean(), 3),
      ", std =", round(task3_values_A.std(), 3),
      ", median =", round(task3_values_A.median(), 3))
print(task3_group_B_name, "-> n =", len(task3_values_B),
      ", mean =", round(task3_values_B.mean(), 3),
      ", std =", round(task3_values_B.std(), 3),
      ", median =", round(task3_values_B.median(), 3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].hist(task3_values_A, bins=12, color="steelblue", edgecolor="white")
axes[0].set_title(task3_group_A_name)
axes[0].set_xlabel(task3_outcome_column)
axes[0].set_ylabel("Number of teams")
axes[1].hist(task3_values_B, bins=12, color="darkorange", edgecolor="white")
axes[1].set_title(task3_group_B_name)
axes[1].set_xlabel(task3_outcome_column)
fig.suptitle("Task 3: Histogram of " + task3_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task3_histogram.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([task3_values_A, task3_values_B],
           tick_labels=[task3_group_A_name, task3_group_B_name])
ax.set_ylabel(task3_outcome_column)
ax.set_title("Task 3: Boxplot of " + task3_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task3_boxplot.png", dpi=150)
plt.close(fig)
print("Saved task3_histogram.png and task3_boxplot.png")

# ---------- STEP 3: 95% CONFIDENCE INTERVAL FOR THE DIFFERENCE IN MEANS ----------

task3_mean_A = task3_values_A.mean()
task3_mean_B = task3_values_B.mean()
task3_mean_difference = task3_mean_A - task3_mean_B

task3_n_A = len(task3_values_A)
task3_n_B = len(task3_values_B)
task3_var_A = task3_values_A.var()
task3_var_B = task3_values_B.var()

task3_standard_error = np.sqrt((task3_var_A / task3_n_A) + (task3_var_B / task3_n_B))

task3_deg_freedom = ((task3_var_A / task3_n_A + task3_var_B / task3_n_B) ** 2) / (
    ((task3_var_A / task3_n_A) ** 2) / (task3_n_A - 1)
    + ((task3_var_B / task3_n_B) ** 2) / (task3_n_B - 1)
)

task3_t_critical = stats.t.ppf(1 - ALPHA / 2, task3_deg_freedom)
task3_margin_of_error = task3_t_critical * task3_standard_error
task3_ci_lower = task3_mean_difference - task3_margin_of_error
task3_ci_upper = task3_mean_difference + task3_margin_of_error

print("\n--- 95% Confidence Interval for the difference in means ---")
print("Mean difference (Above median - Below median):", round(task3_mean_difference, 3))
print("95% CI: [", round(task3_ci_lower, 3), ",", round(task3_ci_upper, 3), "]")

# ---------- STEP 4: TWO-SAMPLE T-TEST ----------

task3_t_statistic, task3_p_value = stats.ttest_ind(
    task3_values_A, task3_values_B, equal_var=False
)

print("\n--- Two-sample t-test (Welch's t-test) ---")
print("t-statistic:", round(task3_t_statistic, 3))
print("p-value:", round(task3_p_value, 5))

if task3_p_value < ALPHA:
    print(f"p-value < {ALPHA}, so we REJECT the null hypothesis.")
    print("Conclusion: there IS a statistically significant difference "
          "between the two groups.")
else:
    print(f"p-value >= {ALPHA}, so we FAIL TO REJECT the null hypothesis.")
    print("Conclusion: we do NOT have enough evidence of a real "
          "difference between the two groups.")


# ======================================================================
# TASK 4: AGE AND PHYSICAL WORKLOAD
# Question: do outfield players under 25 cover more distance per 90
# minutes than outfield players aged 25 or older?
# ======================================================================
print("\n" + "#" * 70)
print("# TASK 4: AGE AND PHYSICAL WORKLOAD")
print("#" * 70)

# ---------- STEP 1: DATA PREPARATION AND SAMPLING ----------

task4_raw = pd.read_csv(f"{UPLOAD_DIR}/task4_real_data.csv", comment="#")

# INCLUSION RULE: outfield players only, with at least 180 minutes
# played, and known age. This file is already clean (no missing
# values), so the population is simply every row in the file.
task4_population = task4_raw

print("Population size:", len(task4_population))
print(task4_population["age_group"].value_counts().to_string())

task4_group_A_name = "Under 25"
task4_group_B_name = "25 and older"

task4_population_A = task4_population[task4_population["age_group"] == task4_group_A_name]
task4_population_B = task4_population[task4_population["age_group"] == task4_group_B_name]

task4_sample_A = task4_population_A.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)
task4_sample_B = task4_population_B.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)

print("Sample size per group:", SAMPLE_SIZE_PER_GROUP)

task4_outcome_column = "distance_km_per90"
task4_values_A = task4_sample_A[task4_outcome_column]
task4_values_B = task4_sample_B[task4_outcome_column]

# ---------- STEP 2: DESCRIPTIVE STATISTICS ----------

print("\n--- Descriptive statistics:", task4_outcome_column, "---")
print(task4_group_A_name, "-> n =", len(task4_values_A),
      ", mean =", round(task4_values_A.mean(), 3),
      ", std =", round(task4_values_A.std(), 3),
      ", median =", round(task4_values_A.median(), 3))
print(task4_group_B_name, "-> n =", len(task4_values_B),
      ", mean =", round(task4_values_B.mean(), 3),
      ", std =", round(task4_values_B.std(), 3),
      ", median =", round(task4_values_B.median(), 3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].hist(task4_values_A, bins=15, color="steelblue", edgecolor="white")
axes[0].set_title(task4_group_A_name)
axes[0].set_xlabel(task4_outcome_column)
axes[0].set_ylabel("Number of players")
axes[1].hist(task4_values_B, bins=15, color="darkorange", edgecolor="white")
axes[1].set_title(task4_group_B_name)
axes[1].set_xlabel(task4_outcome_column)
fig.suptitle("Task 4: Histogram of " + task4_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task4_histogram.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([task4_values_A, task4_values_B],
           tick_labels=[task4_group_A_name, task4_group_B_name])
ax.set_ylabel(task4_outcome_column)
ax.set_title("Task 4: Boxplot of " + task4_outcome_column)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/task4_boxplot.png", dpi=150)
plt.close(fig)
print("Saved task4_histogram.png and task4_boxplot.png")

# ---------- STEP 3: 95% CONFIDENCE INTERVAL FOR THE DIFFERENCE IN MEANS ----------

task4_mean_A = task4_values_A.mean()
task4_mean_B = task4_values_B.mean()
task4_mean_difference = task4_mean_A - task4_mean_B

task4_n_A = len(task4_values_A)
task4_n_B = len(task4_values_B)
task4_var_A = task4_values_A.var()
task4_var_B = task4_values_B.var()

task4_standard_error = np.sqrt((task4_var_A / task4_n_A) + (task4_var_B / task4_n_B))

task4_deg_freedom = ((task4_var_A / task4_n_A + task4_var_B / task4_n_B) ** 2) / (
    ((task4_var_A / task4_n_A) ** 2) / (task4_n_A - 1)
    + ((task4_var_B / task4_n_B) ** 2) / (task4_n_B - 1)
)

task4_t_critical = stats.t.ppf(1 - ALPHA / 2, task4_deg_freedom)
task4_margin_of_error = task4_t_critical * task4_standard_error
task4_ci_lower = task4_mean_difference - task4_margin_of_error
task4_ci_upper = task4_mean_difference + task4_margin_of_error

print("\n--- 95% Confidence Interval for the difference in means ---")
print("Mean difference (Under 25 - 25 and older):", round(task4_mean_difference, 3))
print("95% CI: [", round(task4_ci_lower, 3), ",", round(task4_ci_upper, 3), "]")

# ---------- STEP 4: TWO-SAMPLE T-TEST ----------

task4_t_statistic, task4_p_value = stats.ttest_ind(
    task4_values_A, task4_values_B, equal_var=False
)

print("\n--- Two-sample t-test (Welch's t-test) ---")
print("t-statistic:", round(task4_t_statistic, 3))
print("p-value:", round(task4_p_value, 5))

if task4_p_value < ALPHA:
    print(f"p-value < {ALPHA}, so we REJECT the null hypothesis.")
    print("Conclusion: there IS a statistically significant difference "
          "between the two groups.")
else:
    print(f"p-value >= {ALPHA}, so we FAIL TO REJECT the null hypothesis.")
    print("Conclusion: we do NOT have enough evidence of a real "
          "difference between the two groups.")


# ======================================================================
# FINAL SUMMARY TABLE (handy for pasting straight into your report)
# ======================================================================
print("\n" + "#" * 70)
print("# SUMMARY OF ALL FOUR TASKS")
print("#" * 70)

summary_table = pd.DataFrame({
    "Task": ["Task 1", "Task 2", "Task 3", "Task 4"],
    "Outcome variable": [
        task1_outcome_column, task2_outcome_column,
        task3_outcome_column, task4_outcome_column,
    ],
    "Group A": [task1_group_A_name, task2_group_A_name, task3_group_A_name, task4_group_A_name],
    "Mean A": [task1_mean_A, task2_mean_A, task3_mean_A, task4_mean_A],
    "Group B": [task1_group_B_name, task2_group_B_name, task3_group_B_name, task4_group_B_name],
    "Mean B": [task1_mean_B, task2_mean_B, task3_mean_B, task4_mean_B],
    "Mean difference": [
        task1_mean_difference, task2_mean_difference,
        task3_mean_difference, task4_mean_difference,
    ],
    "95% CI lower": [task1_ci_lower, task2_ci_lower, task3_ci_lower, task4_ci_lower],
    "95% CI upper": [task1_ci_upper, task2_ci_upper, task3_ci_upper, task4_ci_upper],
    "t-statistic": [task1_t_statistic, task2_t_statistic, task3_t_statistic, task4_t_statistic],
    "p-value": [task1_p_value, task2_p_value, task3_p_value, task4_p_value],
})

summary_table = summary_table.round(4)
print(summary_table.to_string(index=False))

summary_table.to_csv(f"{OUTPUT_DIR}/full_analysis_summary.csv", index=False)
print(f"\nSaved full_analysis_summary.csv to {OUTPUT_DIR}")
print("All histograms and boxplots (task1 to task4) have also been saved there.")
