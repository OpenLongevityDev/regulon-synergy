import pandas as pd

expr_path = (
    "/data/analysis/combinations/input/cell_lines/depmap/"
    "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
)

# ------------------------------------------------------------
# Inspect first rows
# ------------------------------------------------------------

x = pd.read_csv(expr_path, nrows=5)

print(x.iloc[:, :6])
print("\nFirst columns:")
print(x.columns[:10])


# ------------------------------------------------------------
# Inspect DepMap model metadata
# ------------------------------------------------------------

meta = pd.read_csv(
    expr_path,
    usecols=[
        "SequencingID",
        "ModelConditionID",
        "ModelID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
    ]
)

print("\n==============================")
print("Basic counts")
print("==============================")

print("Rows:", len(meta))
print("Unique ModelID:", meta["ModelID"].nunique())
print("Unique ModelConditionID:", meta["ModelConditionID"].nunique())
print("Unique SequencingID:", meta["SequencingID"].nunique())


print("\n==============================")
print("IsDefaultEntryForModel")
print("==============================")

print(
    meta["IsDefaultEntryForModel"]
    .value_counts(dropna=False)
)


print("\n==============================")
print("IsDefaultEntryForMC")
print("==============================")

print(
    meta["IsDefaultEntryForMC"]
    .value_counts(dropna=False)
)


print("\n==============================")
print("Number of rows per ModelID")
print("==============================")

print(
    meta["ModelID"]
    .value_counts()
    .value_counts()
    .sort_index()
)


print("\n==============================")
print("Duplicated ModelIDs")
print("==============================")

model_counts = meta["ModelID"].value_counts()

duplicated_models = model_counts[
    model_counts > 1
]

print("Number of ModelIDs with >1 row:", len(duplicated_models))

if len(duplicated_models) > 0:
    print("\nExamples:")
    print(duplicated_models.head(20))

    example_ids = duplicated_models.head(5).index

    print("\nEntries for example duplicated models:")
    print(
        meta[
            meta["ModelID"].isin(example_ids)
        ].sort_values("ModelID")
    )


print("\n==============================")
print("After filtering default models")
print("==============================")

default_meta = meta[
    meta["IsDefaultEntryForModel"] == "Yes"
]

print("Rows:", len(default_meta))
print(
    "Unique ModelID:",
    default_meta["ModelID"].nunique()
)

print(
    "Duplicated ModelIDs remaining:",
    default_meta["ModelID"].duplicated().sum()
)
