"""Post-inference structural validation for output.csv."""
import csv
from collections import Counter
from pathlib import Path

CHALLENGE = Path(__file__).parent.parent
OUTPUT   = CHALLENGE / "output.csv"
CLAIMS   = CHALLENGE / "dataset" / "claims.csv"

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
ALLOWED_VALID_IMAGE  = {"true", "false"}
ALLOWED_ESM          = {"true", "false"}
ALLOWED_SEVERITY     = {"none", "low", "medium", "high", "unknown"}
ALLOWED_ISSUE_TYPE   = {
    "dent","scratch","crack","glass_shatter","broken_part","missing_part",
    "torn_packaging","crushed_packaging","water_damage","stain","none","unknown"
}
ALLOWED_OBJECT_PART = {
    "front_bumper","rear_bumper","door","hood","windshield","side_mirror",
    "headlight","taillight","fender","quarter_panel","body",
    "screen","keyboard","trackpad","hinge","lid","corner","port","base",
    "box","package_corner","package_side","seal","label","contents","item",
    "unknown"
}
ALLOWED_RISK_FLAGS = {
    "none","blurry_image","cropped_or_obstructed","low_light_or_glare",
    "wrong_angle","wrong_object","wrong_object_part","damage_not_visible",
    "claim_mismatch","possible_manipulation","non_original_image",
    "text_instruction_present","user_history_risk","manual_review_required"
}
OUTPUT_COLS_ORDER = [
    "user_id","image_paths","user_claim","claim_object",
    "evidence_standard_met","evidence_standard_met_reason","risk_flags",
    "issue_type","object_part","claim_status","claim_status_justification",
    "supporting_image_ids","valid_image","severity"
]

errors = []

out_rows  = list(csv.DictReader(OUTPUT.open(newline="", encoding="utf-8")))
gold_rows = list(csv.DictReader(CLAIMS.open(newline="", encoding="utf-8")))

# 1: row count
if len(out_rows) != 44:
    errors.append(f"Row count: expected 44, got {len(out_rows)}")
else:
    print(f"[PASS] Row count = {len(out_rows)}")

# 2: column order
actual_cols = list(out_rows[0].keys())
if actual_cols != OUTPUT_COLS_ORDER:
    errors.append(f"Column order mismatch: got {actual_cols}")
else:
    print(f"[PASS] 14 columns in correct order")

# 3: passthrough columns
PASSTHROUGH = ["user_id","image_paths","user_claim","claim_object"]
mismatch_pt = []
for i, (out, gold) in enumerate(zip(out_rows, gold_rows)):
    for col in PASSTHROUGH:
        if out.get(col) != gold.get(col):
            mismatch_pt.append(f"  row {i+1} col={col!r}: out={out[col]!r} gold={gold[col]!r}")
if mismatch_pt:
    errors.append("Passthrough column mismatches:\n" + "\n".join(mismatch_pt[:5]))
else:
    print(f"[PASS] All 4 passthrough columns match claims.csv row-for-row")

# 4: enum validation
enum_errors = []
for i, row in enumerate(out_rows, 1):
    uid = row["user_id"]
    prefix = f"  row {i:02d} ({uid}):"
    if row["claim_status"] not in ALLOWED_CLAIM_STATUS:
        enum_errors.append(f"{prefix} claim_status={row['claim_status']!r}")
    if row["valid_image"] not in ALLOWED_VALID_IMAGE:
        enum_errors.append(f"{prefix} valid_image={row['valid_image']!r}")
    if row["evidence_standard_met"] not in ALLOWED_ESM:
        enum_errors.append(f"{prefix} esm={row['evidence_standard_met']!r}")
    if row["severity"] not in ALLOWED_SEVERITY:
        enum_errors.append(f"{prefix} severity={row['severity']!r}")
    if row["issue_type"] not in ALLOWED_ISSUE_TYPE:
        enum_errors.append(f"{prefix} issue_type={row['issue_type']!r}")
    for part in row["object_part"].split(";"):
        if part.strip() not in ALLOWED_OBJECT_PART:
            enum_errors.append(f"{prefix} object_part={part.strip()!r}")
    for flag in row["risk_flags"].split(";"):
        if flag.strip() not in ALLOWED_RISK_FLAGS:
            enum_errors.append(f"{prefix} risk_flag={flag.strip()!r}")

if enum_errors:
    errors.append("Enum violations:\n" + "\n".join(enum_errors))
else:
    print(f"[PASS] All enums valid across 44 rows")

# 5: none must not mix with real values
none_mix_errors = []
for i, row in enumerate(out_rows, 1):
    uid = row["user_id"]
    flags = [f.strip() for f in row["risk_flags"].split(";")]
    if "none" in flags and len(flags) > 1:
        none_mix_errors.append(f"  row {i:02d} ({uid}): risk_flags has 'none' + real flags: {flags}")
    ids = [x.strip() for x in row["supporting_image_ids"].split(";")]
    if "none" in ids and len(ids) > 1:
        none_mix_errors.append(f"  row {i:02d} ({uid}): supporting_image_ids 'none' + real: {ids}")

if none_mix_errors:
    errors.append("'none' mixed with real values:\n" + "\n".join(none_mix_errors))
else:
    print(f"[PASS] 'none' never alongside real flags/IDs")

# 6: supporting_image_ids reference submitted images
id_errors = []
for i, row in enumerate(out_rows, 1):
    uid = row["user_id"]
    image_paths = row["image_paths"].split(";")
    allowed_ids = {Path(p.strip()).stem for p in image_paths}
    supp_raw = row["supporting_image_ids"].strip()
    if supp_raw == "none":
        continue
    for sid in supp_raw.split(";"):
        sid = sid.strip()
        if sid and sid != "none" and sid not in allowed_ids:
            id_errors.append(f"  row {i:02d} ({uid}): id={sid!r} not in {sorted(allowed_ids)}")

if id_errors:
    errors.append("Supporting ID outside submitted images:\n" + "\n".join(id_errors))
else:
    print(f"[PASS] All supporting_image_ids reference submitted images")

# 7: repeated user_ids all preserved
uid_counts = Counter(r["user_id"] for r in out_rows)
dupes = {k: v for k, v in uid_counts.items() if v > 1}
if dupes:
    print(f"[PASS] Repeated user_ids preserved: {dupes}")
print(f"[PASS] Total rows={len(out_rows)}, unique user_ids={len(uid_counts)}")

# Distributions
print(f"[INFO] claim_status: {dict(Counter(r['claim_status'] for r in out_rows))}")
print(f"[INFO] valid_image:  {dict(Counter(r['valid_image'] for r in out_rows))}")
print(f"[INFO] esm:          {dict(Counter(r['evidence_standard_met'] for r in out_rows))}")

# Manual-review flags: rows needing human attention
mr_rows = [r for r in out_rows if "manual_review_required" in r["risk_flags"]]
ti_rows = [r for r in out_rows if "text_instruction_present" in r["risk_flags"]]
ni_rows = [r for r in out_rows if "non_original_image" in r["risk_flags"]]
pm_rows = [r for r in out_rows if "possible_manipulation" in r["risk_flags"]]
nei_rows = [r for r in out_rows if r["claim_status"] == "not_enough_information"]
inv_rows = [r for r in out_rows if r["valid_image"] == "false"]

print(f"[INFO] manual_review_required flags: {len(mr_rows)} rows")
print(f"[INFO] text_instruction_present:     {len(ti_rows)} rows")
print(f"[INFO] non_original_image:           {len(ni_rows)} rows")
print(f"[INFO] possible_manipulation:        {len(pm_rows)} rows")
print(f"[INFO] NEI (not_enough_info) rows:   {len(nei_rows)}")
print(f"[INFO] valid_image=false rows:       {len(inv_rows)}")

print()
if errors:
    print(f"=== VALIDATION FAILED ({len(errors)} error groups) ===")
    for e in errors:
        print(e)
else:
    print("=== VALIDATION PASSED - all checks green ===")
