"""local054 - customers spending under $1 on the best-selling artist's albums.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Best-selling artist = highest total album sales; per-customer spend on that
artist's tracks; keep those below $1.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local054"

REFERENCE_SQL = """
WITH artist_sales AS (
    SELECT ar.ArtistId, SUM(ii.UnitPrice * ii.Quantity) AS s
    FROM invoice_items ii
    JOIN tracks t ON ii.TrackId = t.TrackId
    JOIN albums al ON t.AlbumId = al.AlbumId
    JOIN artists ar ON al.ArtistId = ar.ArtistId
    GROUP BY ar.ArtistId ORDER BY s DESC LIMIT 1
)
SELECT c.FirstName, SUM(ii.UnitPrice * ii.Quantity) AS TOTALSPENT
FROM customers c
JOIN invoices i ON c.CustomerId = i.CustomerId
JOIN invoice_items ii ON i.InvoiceId = ii.InvoiceId
JOIN tracks t ON ii.TrackId = t.TrackId
JOIN albums al ON t.AlbumId = al.AlbumId
WHERE al.ArtistId IN (SELECT ArtistId FROM artist_sales)
GROUP BY c.CustomerId
HAVING TOTALSPENT < 1
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_ii = import_table(reg, rec, IID, "invoice_items", ["InvoiceId", "TrackId", "UnitPrice", "Quantity"])
    m_inv = import_table(reg, rec, IID, "invoices", ["InvoiceId", "CustomerId"], index=["InvoiceId"])
    m_trk = import_table(reg, rec, IID, "tracks", ["TrackId", "AlbumId"], index=["TrackId"])
    m_alb = import_table(reg, rec, IID, "albums", ["AlbumId", "ArtistId"], index=["AlbumId"])
    m_art = import_table(reg, rec, IID, "artists", ["ArtistId", "Name"])
    m_cus = import_table(reg, rec, IID, "customers", ["CustomerId", "FirstName"], index=["CustomerId"])
    coll = reg.db[m_ii["collection"]]

    PIPELINE = [
        {"$lookup": {"from": m_inv["collection"], "localField": "InvoiceId",
                     "foreignField": "InvoiceId", "as": "inv"}},
        {"$unwind": "$inv"},
        {"$lookup": {"from": m_trk["collection"], "localField": "TrackId",
                     "foreignField": "TrackId", "as": "t"}},
        {"$unwind": "$t"},
        {"$lookup": {"from": m_alb["collection"], "localField": "t.AlbumId",
                     "foreignField": "AlbumId", "as": "al"}},
        {"$unwind": "$al"},
        {"$addFields": {"cust": "$inv.CustomerId", "artist": "$al.ArtistId",
                        "spend": {"$multiply": ["$UnitPrice", "$Quantity"]}}},
        {"$facet": {
            "best": [{"$group": {"_id": "$artist", "s": {"$sum": "$spend"}}},
                     {"$sort": {"s": -1}}, {"$limit": 1}],
            "per": [{"$group": {"_id": {"c": "$cust", "a": "$artist"}, "s": {"$sum": "$spend"}}},
                    {"$project": {"_id": 0, "c": "$_id.c", "a": "$_id.a", "s": 1}}]}},
        {"$unwind": "$best"},
        {"$project": {"bestId": "$best._id", "per": 1}},
        {"$unwind": "$per"},
        {"$match": {"$expr": {"$eq": ["$per.a", "$bestId"]}}},
        {"$group": {"_id": "$per.c", "TOTALSPENT": {"$sum": "$per.s"}}},
        {"$match": {"TOTALSPENT": {"$lt": 1}}},
        {"$lookup": {"from": m_cus["collection"], "localField": "_id",
                     "foreignField": "CustomerId", "as": "c"}},
        {"$unwind": "$c"},
        {"$project": {"_id": 0, "FirstName": "$c.FirstName", "TOTALSPENT": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["FirstName", "TOTALSPENT"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "global top-1 (facet) then per-customer filtered sum",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Best-selling artist = max total album sales; per-customer spend on that artist's tracks; keep spend < 1."),
        "source_schema": {"invoice_items": ["InvoiceId", "TrackId", "UnitPrice", "Quantity"],
                          "invoices": ["InvoiceId", "CustomerId"], "tracks": ["TrackId", "AlbumId"],
                          "albums": ["AlbumId", "ArtistId"], "artists": ["ArtistId", "Name"],
                          "customers": ["CustomerId", "FirstName"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_ii, m_inv, m_trk, m_alb, m_art, m_cus],
                    "_id": "default ObjectId", "indexes": ["invoices.InvoiceId", "tracks.TrackId", "albums.AlbumId", "customers.CustomerId"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Best-selling artist is determined by highest total sales (sum of unitprice*quantity over all invoice items).",
            "Spend is computed only on that artist's albums; HAVING is applied after aggregation.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
