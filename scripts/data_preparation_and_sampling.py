"""
HIT140 Group Project - FIFA World Cup 2026
Data Preparation and Sampling (all four tasks)

What this script does, in order, for each task:
  1. Load the wrangled CSV and skip the metadata comment lines.
  2. Define the population and apply/verify the inclusion criteria stated
     in the Question Formulation document.
  3. Work out how large a sample needs to be for a two-sample t-test with
     a reasonable chance of detecting a real effect (power analysis).
  4. Draw the sample with an appropriate technique (stratified random
     sampling when the population is large enough, full census when it
     is not) and save it for the descriptive/inferential steps that come
     after this.

Run this once. It writes four *_sample.csv files and prints the
population/sample summary you need for the report.

Note on two issues found while preparing this data (read before you use
the outputs):

  - Task 1: two players (Lionel Messi, Kevin Pina) have a missing
    "position" value. The inclusion criterion requires outfield players
    only, and position cannot be confirmed for these two rows, so they
    are dropped. This brings the population from 484 to 482, which
    matches the population size already used in Task 4 for the same two
    players (dropped there for missing date of birth). The two datasets
    are now consistent, but the raw Task 1 file was not clean on
    delivery.

  - Task 2: the file has only 10 matches, not the full set of matches
    played at the tournament, and they look hand-picked (four of the ten
    involve Argentina) rather than randomly selected. Ten matches is
    also far short of the number needed for a two-sample t-test to have
    any real power. This script uses all 10 as a census because there is
    nothing else to sample from, but this is a genuine weakness: the
    conclusion for Task 2 will rest on a small, non-random convenience
    sample, and that limitation needs to be stated plainly in the
    report, not glossed over. The fix is to pull the complete match list
    (104 matches) and re-run the sampling step below.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

RANDOM_SEED = 42
ALPHA = 0.05
POWER = 0.80
ASSUMED_EFFECT_SIZE = 0.5  # medium effect (Cohen's d); no pilot data exists
                           # to base this on, so a medium effect is the
                           # standard default assumption. State this
                           # assumption in the report.

UPLOAD_DIR = "data"
OUTPUT_DIR = "outputs"

rng = np.random.default_rng(RANDOM_SEED)


def required_n_per_group(effect_size=ASSUMED_EFFECT_SIZE, alpha=ALPHA, power=POWER):
    """Minimum sample size per group for an independent two-sample t-test."""
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=1,
        alternative="two-sided",
    )
    return int(np.ceil(n))


def stratified_sample(df, group_col, n_per_group, seed=RANDOM_SEED):
    """
    Draw up to n_per_group rows at random from each level of group_col,
    without replacement. If a group has fewer rows than n_per_group,
    every row in that group is kept (you cannot sample more than exists).
    """
    parts = []
    for level, group_df in df.groupby(group_col):
        take = min(n_per_group, len(group_df))
        parts.append(group_df.sample(n=take, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def report(task_name, population_df, sample_df, group_col):
    print(f"\n=== {task_name} ===")
    print(f"Population size: {len(population_df)}")
    print("Population by group:")
    print(population_df[group_col].value_counts().to_string())
    print(f"Sample size: {len(sample_df)}")
    print("Sample by group:")
    print(sample_df[group_col].value_counts().to_string())


# ---------------------------------------------------------------------
# TASK 1 - Defensive Work Rate by Tournament Stage (unit: player)
# ---------------------------------------------------------------------
t1 = pd.read_csv(f"{UPLOAD_DIR}/task1_real_data_1.csv", comment="#")

# Inclusion criterion: outfield players only, >=180 minutes played.
# The minutes filter is already applied in the file. Position cannot be
# confirmed as outfield for 2 rows with a missing value, so drop them.
t1_population = t1.dropna(subset=["position"]).copy()
assert (t1_population["minutes_played"] >= 180).all()
assert t1_population["position"].isin(["D", "M", "F"]).all()

n_req_1 = required_n_per_group()
t1_sample = stratified_sample(t1_population, "tournament_stage", n_req_1)
report("Task 1: Defensive Work Rate", t1_population, t1_sample, "tournament_stage")

# ---------------------------------------------------------------------
# TASK 2 - Disciplinary Behaviour Across Match Stages (unit: match)
# ---------------------------------------------------------------------
t2 = pd.read_csv(f"{UPLOAD_DIR}/task2_real_data_csvArik.csv", comment="#")
t2_population = t2.copy()  # inclusion criterion: all matches, none abandoned/forfeited

# Population is only 10 matches and was not assembled by random selection
# (see the note at the top of this file). There is no larger pool here to
# sample from, so the census (all 10 rows) is carried forward as the
# "sample". Flag this as a limitation in the report rather than treating
# it as equivalent to a proper random sample.
t2_sample = t2_population.copy()
report("Task 2: Match Discipline", t2_population, t2_sample, "knockout_stage")
print("LIMITATION: n=10, non-random selection. Statistical power will be low "
      "and results should be treated as exploratory, not conclusive.")

# ---------------------------------------------------------------------
# TASK 3 - Possession and Shooting Discipline (unit: team)
# ---------------------------------------------------------------------
t3 = pd.read_csv(f"{UPLOAD_DIR}/task3_final_wrangled_csvRatul.csv", comment="#")
t3_population = t3.dropna(subset=["Poss", "SoTpct"]).copy()  # drop incomplete records

# Population is the 48 participating teams - already smaller than the
# ~64-per-group figure a power analysis would ask for. There is no
# larger population of "teams at this tournament" to sample from, so
# take the full census rather than throwing away data by subsampling.
t3_sample = t3_population.copy()
report("Task 3: Possession and Shooting", t3_population, t3_sample, "group")
print(f"NOTE: population (48 teams) is below the {required_n_per_group()}-per-group "
      "figure a medium-effect power analysis calls for. Using the full census is "
      "the right call here, but treat any non-significant result with caution - "
      "it may reflect low power rather than a genuine absence of an effect.")

# ---------------------------------------------------------------------
# TASK 4 - Age and Physical Workload (unit: player)
# ---------------------------------------------------------------------
t4 = pd.read_csv(f"{UPLOAD_DIR}/task4_real_data.csv", comment="#")
t4_population = t4.copy()  # minutes filter and DOB exclusion already applied (n=482)
assert (t4_population["minutes_played"] >= 180).all()
assert t4_population["position"].isin(["D", "M", "F"]).all()

n_req_4 = required_n_per_group()
t4_sample = stratified_sample(t4_population, "age_group", n_req_4)
report("Task 4: Age and Physical Workload", t4_population, t4_sample, "age_group")

# ---------------------------------------------------------------------
# Save the sampled datasets for the next steps (descriptive stats, CI, t-test)
# ---------------------------------------------------------------------
t1_sample.to_csv(f"{OUTPUT_DIR}/task1_sample.csv", index=False)
t2_sample.to_csv(f"{OUTPUT_DIR}/task2_sample.csv", index=False)
t3_sample.to_csv(f"{OUTPUT_DIR}/task3_sample.csv", index=False)
t4_sample.to_csv(f"{OUTPUT_DIR}/task4_sample.csv", index=False)

print(f"\nRequired n per group for a two-sample t-test "
      f"(d={ASSUMED_EFFECT_SIZE}, alpha={ALPHA}, power={POWER}): {n_req_1} per group")
print("Saved: task1_sample.csv, task2_sample.csv, task3_sample.csv, task4_sample.csv")
