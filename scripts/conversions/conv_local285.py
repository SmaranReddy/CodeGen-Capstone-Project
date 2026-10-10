"""local285 - vegetable wholesale financial performance per category x year (2020-2023).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Sales join the category map on item_code and the daily wholesale price on
(item_code, date) and the item loss rate on item_code. Per (year, category):
avg/max/min wholesale price, their difference, sale-only total wholesale/selling price,
average loss rate, sale-only total loss (qty*whsle*loss/100) and profit. Reproduces gold
variant `_a` (cols [2..9]).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local285"
W = "whsle_px_rmb-kg"
Q = "qty_sold(kg)"
SP = "unit_selling_px_rmb/kg"
SR = "sale/return"
LR = "loss_rate_%"

REFERENCE_SQL = f"""
WITH t AS (
  SELECT c.category_name AS cat, substr(t.txn_date,1,4) AS yr,
         t."{Q}" AS q, t."{SP}" AS sp, t."{SR}" AS sr,
         w."{W}" AS wpx, l."{LR}" AS lr
  FROM veg_txn_df t
  JOIN veg_cat c ON c.item_code = t.item_code
  JOIN veg_whsle_df w ON w.item_code = t.item_code AND w.whsle_date = t.txn_date
  JOIN veg_loss_rate_df l ON l.item_code = t.item_code)
SELECT yr AS YEAR, cat AS category_name,
       AVG(wpx) AS AVG_WHOLESALE_PRICE,
       MAX(wpx) AS MAX_WHOLESALE_PRICE,
       MIN(wpx) AS MIN_WHOLESALE_PRICE,
       MAX(wpx) - MIN(wpx) AS WHOLESALE_PRICE_DIFFERENCE,
       SUM(CASE WHEN sr='sale' THEN q*wpx ELSE 0 END) AS TOTAL_WHOLESALE_PRICE,
       SUM(CASE WHEN sr='sale' THEN q*sp ELSE 0 END) AS TOTAL_SELLING_PRICE,
       AVG(lr) AS AVG_LOSS_RATE,
       SUM(CASE WHEN sr='sale' THEN q*wpx*lr/100 ELSE 0 END) AS TOTAL_LOSS
FROM t GROUP BY yr, cat
"""

KEY = {"$concat": [{"$toString": "$item_code"}, "|", "$txn_date"]}

PIPELINE = [
    {"$addFields": {"key": KEY}},
    {"$lookup": {"from": "__WKEY__", "localField": "key", "foreignField": "key", "as": "w"}},
    {"$unwind": "$w"},
    {"$lookup": {"from": "__CAT__", "localField": "item_code", "foreignField": "item_code", "as": "c"}},
    {"$unwind": "$c"},
    {"$lookup": {"from": "__LOSS__", "localField": "item_code", "foreignField": "item_code", "as": "l"}},
    {"$unwind": "$l"},
    {"$group": {
        "_id": {"yr": {"$substrCP": ["$txn_date", 0, 4]}, "cat": "$c.category_name"},
        "AVG_WHOLESALE_PRICE": {"$avg": f"$w.{W}"},
        "MAX_WHOLESALE_PRICE": {"$max": f"$w.{W}"},
        "MIN_WHOLESALE_PRICE": {"$min": f"$w.{W}"},
        "TOTAL_WHOLESALE_PRICE": {"$sum": {"$cond": [{"$eq": [f"${SR}", "sale"]},
                                                     {"$multiply": [f"${Q}", f"$w.{W}"]}, 0]}},
        "TOTAL_SELLING_PRICE": {"$sum": {"$cond": [{"$eq": [f"${SR}", "sale"]},
                                                   {"$multiply": [f"${Q}", f"${SP}"]}, 0]}},
        "AVG_LOSS_RATE": {"$avg": f"$l.{LR}"},
        "TOTAL_LOSS": {"$sum": {"$cond": [{"$eq": [f"${SR}", "sale"]},
                                          {"$divide": [{"$multiply": [f"${Q}", f"$w.{W}", f"$l.{LR}"]}, 100]}, 0]}},
    }},
    {"$project": {"_id": 0, "YEAR": {"$toInt": "$_id.yr"}, "category_name": "$_id.cat",
                  "AVG_WHOLESALE_PRICE": 1, "MAX_WHOLESALE_PRICE": 1, "MIN_WHOLESALE_PRICE": 1,
                  "WHOLESALE_PRICE_DIFFERENCE": {"$subtract": ["$MAX_WHOLESALE_PRICE", "$MIN_WHOLESALE_PRICE"]},
                  "TOTAL_WHOLESALE_PRICE": 1, "TOTAL_SELLING_PRICE": 1, "AVG_LOSS_RATE": 1, "TOTAL_LOSS": 1,
                  "PROFIT": {"$subtract": [{"$subtract": ["$TOTAL_SELLING_PRICE", "$TOTAL_WHOLESALE_PRICE"]},
                                           "$TOTAL_LOSS"]}}},
]

COLS = ["YEAR", "category_name", "AVG_WHOLESALE_PRICE", "MAX_WHOLESALE_PRICE", "MIN_WHOLESALE_PRICE",
        "WHOLESALE_PRICE_DIFFERENCE", "TOTAL_WHOLESALE_PRICE", "TOTAL_SELLING_PRICE", "AVG_LOSS_RATE",
        "TOTAL_LOSS", "PROFIT"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref["PROFIT"] = ref["TOTAL_SELLING_PRICE"] - ref["TOTAL_WHOLESALE_PRICE"] - ref["TOTAL_LOSS"]
    ref_v, ref_ok = score(ref, rec)

    txn = import_table(reg, rec, IID, "veg_txn_df", ["item_code", "txn_date", Q, SP, SR])
    whsle = import_table(reg, rec, IID, "veg_whsle_df", ["item_code", "whsle_date", W])
    cat = import_table(reg, rec, IID, "veg_cat", ["item_code", "category_name"], index=["item_code"])
    loss = import_table(reg, rec, IID, "veg_loss_rate_df", ["item_code", LR], index=["item_code"])

    wkey = f"{IID}__whsle_keyed"
    reg.dropped_first(wkey)
    reg.db[whsle["collection"]].aggregate([
        {"$project": {"_id": 0, "key": {"$concat": [{"$toString": "$item_code"}, "|", "$whsle_date"]}, W: f"${W}"}},
        {"$out": wkey},
    ])
    reg.db[wkey].create_index("key")
    reg.note(wkey)

    mapping = {"__WKEY__": wkey, "__CAT__": cat["collection"], "__LOSS__": loss["collection"]}
    coll = reg.db[txn["collection"]]
    pipeline = [{**s, "$lookup": {**s["$lookup"], "from": mapping[s["$lookup"]["from"]]}} if "$lookup" in s else s
                for s in PIPELINE]
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "multi-join fact aggregation per category-year (avg/max/min/sums) with composite-key $lookup",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per category x year: wholesale price avg/max/min and range; sale-only total wholesale "
                                 "(qty*whsle), total selling (qty*sell), total loss (qty*whsle*loss/100); average loss "
                                 "rate; profit = selling - wholesale - loss."),
        "source_schema": {"veg_txn_df": ["item_code", "txn_date", Q, SP, SR],
                          "veg_whsle_df": ["item_code", "whsle_date", W],
                          "veg_cat": ["item_code", "category_name"],
                          "veg_loss_rate_df": ["item_code", LR]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [txn, whsle, cat, loss],
                    "derived": [{"collection": wkey, "method": "$out of composite item|date key"}]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Wholesale price joined on the exact (item_code, txn_date); loss rate on item_code.",
                        "Returns carry negative quantities; sale-only sums use 'sale' rows explicitly.",
                        "avg/max/min use all joined rows (the gold variant is not consistent across its four files).",
                        "Aggregates are left unrounded (values differ from gold's 2-dp rounding by <=0.005, within tolerance).",
                        "Validated against gold variant _a (cols [2..9])."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
