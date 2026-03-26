import kingfisher_bnb
import numpy as np

# 1. Create dummy data
# 0: Apple, 1: Banana, 2: Cherry, 3: Diet
# Let's say we only care about 'Diet' (3) as a consequence.
# And we want to exclude Banana (1) being a cause for Diet (3) for some reason.
data = [
    [0, 1, 3],
    [0, 3],
    [1, 2, 3],
    [1, 3],
    [0, 1, 2],
    [0, 2],
    [3],
    [0, 1, 3],
    [1, 3]
]

names = ["Apple", "Banana", "Cherry", "Diet"]
k = len(names) - 1

print("--- Running Kingfisher with filters ---")

# 2. Find rules with constraints:
# - only 'Diet' as consequent (consequent_only=[3])
# - exclude 'Banana' -> 'Diet' (constraints=[(1, 3)])
# - exclude 'Cherry' from the whole search (excluded_attributes=[2])
rules = kingfisher_bnb.find_rules_from_data(
    data=data,
    k=k,
    q=10,
    l_max=3,
    m_threshold=1.0, # High alpha for demo
    consequent_only=[3],
    constraints=[(1, 3)],
    excluded_attributes=[2]
)

print(f"Found {len(rules)} rules matching constraints:")
for r in rules:
    ant = [names[i] for i in r.antecedent]
    cons = names[r.consequent]
    sign = "NOT " if r.is_negative else ""
    print(f"IF {ant} THEN {sign}{cons} (p={np.exp(r.measure_value):.4f})")

# Verification:
# Rule 'Banana' -> 'Diet' should be missing.
# Rule 'Cherry' -> ... should be missing.
# Only 'Diet' should be the consequent.
for r in rules:
    assert r.consequent == 3, f"Unexpected consequent: {r.consequent}"
    assert 2 not in r.antecedent, "Cherry (2) should be excluded from antecedents"
    if not r.is_negative:
        assert not (1 in r.antecedent and r.consequent == 3), "Constraint (1, 3) violated"

print("\nAll constraints verified!")
