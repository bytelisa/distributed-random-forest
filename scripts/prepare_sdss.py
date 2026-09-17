import pandas as pd

# Simple preprocessing before uploading Sloan Survey Stellar Classification Dataset
# the raw SDSS DR14 dump has identifier columns (objid,
# specobjid, run, rerun, camcol, field, plate, mjd, fiberid) that are numeric
# but not physical features - prepare_features would keep them as-is, risking
# overfitting on IDs. This keeps only the physically meaningful columns.

KEEP_COLUMNS = ["ra", "dec", "u", "g", "r", "i", "z", "redshift", "class"]

df = pd.read_csv("data/sloan_stellar_classification.csv")
df = df[KEEP_COLUMNS]
df.to_csv("data/sdss.csv", index=False)

print(f"[PrepareSDSS] {df.shape[0]} rows, {df.shape[1]} columns -> data/sdss.csv")
print(f"[PrepareSDSS] class distribution:\n{df['class'].value_counts()}")
