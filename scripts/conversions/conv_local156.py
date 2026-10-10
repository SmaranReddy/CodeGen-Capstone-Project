"""local156 - annual average BTC purchase price per region (bank_sales_trading).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. For each region-year: avg purchase price = SUM(quantity*price)/SUM(quantity)
over Bitcoin BUY transactions (price = BTC market price on the txn date, matched by
ticker+date). Regions are ranked per year and the year-over-year % change is computed
against the prior year (2017 is used as the base, then excluded from output).
Reproduces gold variant `_a` (0-based cols [0,1,2,4]).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local156"

REFERENCE_SQL = """
WITH base AS (
  SELECT CAST(substr(t.txn_date,7,4) AS INTEGER) AS yr, m.region,
         SUM(t.quantity*p.price) AS spent, SUM(t.quantity) AS qty
  FROM bitcoin_transactions t
  JOIN bitcoin_members m ON m.member_id = t.member_id
  JOIN bitcoin_prices p ON p.ticker = t.ticker AND p.market_date = t.txn_date
  WHERE t.txn_type = 'BUY' AND t.ticker = 'BTC'
  GROUP BY yr, m.region),
avgp AS (SELECT yr, region, spent/qty AS avg_price FROM base),
seq AS (SELECT yr, region, avg_price,
          (avg_price - LAG(avg_price) OVER (PARTITION BY region ORDER BY yr)) /
          LAG(avg_price) OVER (PARTITION BY region ORDER BY yr) * 100 AS pct
        FROM avgp)
SELECT yr AS txn_year, region, avg_price AS avg_purchase_price,
       RANK() OVER (PARTITION BY yr ORDER BY avg_price DESC) AS avg_price_rank,
       pct AS pct_change_from_prior_year
FROM seq WHERE yr <> 2017 ORDER BY yr, avg_price DESC
"""

PIPELINE_A = [
    {"$lookup": {"from": "__MEMBERS__", "localField": "member_id", "foreignField": "member_id", "as": "m"}},
    {"$unwind": "$m"},
    {"$lookup": {"from": "__PRICES__", "localField": "txn_date", "foreignField": "market_date", "as": "p"}},
    {"$unwind": "$p"},
    {"$addFields": {"yr": {"$toInt": {"$substrCP": ["$txn_date", 6, 4]}},
                    "spent": {"$multiply": ["$quantity", "$p.price"]}}},
    {"$group": {"_id": {"region": "$m.region", "yr": "$yr"},
                "spent": {"$sum": "$spent"}, "qty": {"$sum": "$quantity"}}},
    {"$addFields": {"region": "$_id.region", "yr": "$_id.yr", "avg": {"$divide": ["$spent", "$qty"]}}},
]

PIPELINE_B = [
    {"$setWindowFields": {"partitionBy": "$region", "sortBy": {"yr": 1},
                          "output": {"prev": {"$shift": {"output": "$avg", "by": -1, "default": None}}}}},
    {"$addFields": {"pct": {"$multiply": [{"$divide": [{"$subtract": ["$avg", "$prev"]}, "$prev"]}, 100]}}},
    {"$match": {"yr": {"$ne": 2017}}},
    {"$setWindowFields": {"partitionBy": "$yr", "sortBy": {"avg": -1},
                          "output": {"rank": {"$rank": {}}}}},
    {"$project": {"_id": 0, "txn_year": "$yr", "region": 1, "avg_purchase_price": "$avg",
                  "avg_price_rank": "$rank", "pct_change_from_prior_year": "$pct"}},
]

COLS = ["txn_year", "region", "avg_purchase_price", "avg_price_rank", "pct_change_from_prior_year"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    txns = import_table(reg, rec, IID, "bitcoin_transactions",
                        ["member_id", "txn_date", "quantity", "ticker", "txn_type"],
                        where="txn_type = 'BUY' AND ticker = 'BTC'", index=["member_id"])
    members = import_table(reg, rec, IID, "bitcoin_members", ["member_id", "region"], index=["member_id"])
    prices = import_table(reg, rec, IID, "bitcoin_prices", ["ticker", "market_date", "price"],
                          where="ticker = 'BTC'", index=["market_date"])

    coll = reg.db[txns["collection"]]

    def subst(stage):
        s = dict(stage)
        if "$lookup" in s:
            lk = dict(s["$lookup"])
            if lk["from"] == "__MEMBERS__":
                lk["from"] = members["collection"]
            elif lk["from"] == "__PRICES__":
                lk["from"] = prices["collection"]
            s["$lookup"] = lk
        return s

    pipeline = [subst(s) for s in PIPELINE_A] + PIPELINE_B
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "multi-table join aggregated to region-year + window yoy % + per-year rank",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("avg purchase price per BTC = SUM(quantity*price)/SUM(quantity) over BTC BUY transactions, "
                                 "per region-year; price matched to the BTC market price on the transaction date; "
                                 "year-over-year % change vs prior year (2017 used as base, excluded from output)."),
        "source_schema": {"bitcoin_transactions": ["member_id", "txn_date", "quantity", "ticker", "txn_type"],
                          "bitcoin_members": ["member_id", "region"],
                          "bitcoin_prices": ["ticker", "market_date", "price"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [txns, members, prices], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Only Bitcoin (ticker='BTC') BUY transactions count as purchases.",
                        "Price joined on the exact calendar date; region from bitcoin_members.",
                        "2017 is computed to serve as the prior-year base but excluded from output rows.",
                        "Validated against gold variant _a (cols [0,1,2,4])."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
