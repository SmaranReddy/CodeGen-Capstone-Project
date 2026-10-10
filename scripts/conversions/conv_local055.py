"""local055 - absolute difference of average spending on the best and worst selling artists.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Best/worst albums-selling artists break ties by artist name; per-customer spend
is computed for each, and the two average spends are differenced.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local055"

REFERENCE_SQL = """
WITH asales AS (
    SELECT ar.ArtistId, ar.Name, SUM(ii.UnitPrice * ii.Quantity) AS s
    FROM invoice_items ii
    JOIN tracks t ON ii.TrackId = t.TrackId
    JOIN albums al ON t.AlbumId = al.AlbumId
    JOIN artists ar ON al.ArtistId = ar.ArtistId
    GROUP BY ar.ArtistId, ar.Name
),
top AS (SELECT ArtistId FROM asales ORDER BY s DESC, Name ASC LIMIT 1),
bot AS (SELECT ArtistId FROM asales ORDER BY s ASC, Name ASC LIMIT 1),
spend AS (
    SELECT al.ArtistId AS aid, c.CustomerId AS cid, SUM(ii.UnitPrice * ii.Quantity) AS sp
    FROM customers c
    JOIN invoices i ON c.CustomerId = i.CustomerId
    JOIN invoice_items ii ON i.InvoiceId = ii.InvoiceId
    JOIN tracks t ON ii.TrackId = t.TrackId
    JOIN albums al ON t.AlbumId = al.AlbumId
    GROUP BY al.ArtistId, c.CustomerId
)
SELECT ABS(AVG(CASE WHEN aid IN (SELECT ArtistId FROM top) THEN sp END)
         - AVG(CASE WHEN aid IN (SELECT ArtistId FROM bot) THEN sp END)) AS AbsoluteAverageDifference
FROM spend
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_ii = import_table(reg, rec, IID, "invoice_items", ["InvoiceId", "TrackId", "UnitPrice", "Quantity"])
    m_inv = import_table(reg, rec, IID, "invoices", ["InvoiceId", "CustomerId"], index=["InvoiceId"])
    m_trk = import_table(reg, rec, IID, "tracks", ["TrackId", "AlbumId"], index=["TrackId"])
    m_alb = import_table(reg, rec, IID, "albums", ["AlbumId", "ArtistId"], index=["AlbumId"])
    m_art = import_table(reg, rec, IID, "artists", ["ArtistId", "Name"], index=["ArtistId"])
    coll = reg.db[m_ii["collection"]]

    enrich = [
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
    ]
    PIPELINE = enrich + [
        {"$facet": {
            "a": [{"$group": {"_id": "$artist", "s": {"$sum": "$spend"}}},
                  {"$lookup": {"from": m_art["collection"], "localField": "_id",
                               "foreignField": "ArtistId", "as": "ar"}},
                  {"$unwind": "$ar"},
                  {"$sort": {"s": 1, "ar.Name": 1}},
                  {"$group": {"_id": None, "botId": {"$first": "$_id"}, "topId": {"$last": "$_id"}}}],
            "per": [{"$group": {"_id": {"c": "$cust", "a": "$artist"}, "s": {"$sum": "$spend"}}}]}},
        {"$project": {"topId": {"$arrayElemAt": ["$a.topId", 0]},
                      "botId": {"$arrayElemAt": ["$a.botId", 0]}, "per": 1}},
        {"$unwind": "$per"},
        {"$addFields": {"which": {"$switch": {"branches": [
            {"case": {"$eq": ["$per._id.a", "$topId"]}, "then": "top"},
            {"case": {"$eq": ["$per._id.a", "$botId"]}, "then": "bot"}], "default": "other"}}}},
        {"$match": {"which": {"$ne": "other"}}},
        {"$group": {"_id": "$which", "avg": {"$avg": "$per.s"}}},
        {"$group": {"_id": None,
                    "top": {"$max": {"$cond": [{"$eq": ["$_id", "top"]}, "$avg", -1]}},
                    "bot": {"$max": {"$cond": [{"$eq": ["$_id", "bot"]}, "$avg", -1]}}}},
        {"$project": {"_id": 0, "AbsoluteAverageDifference": {"$abs": {"$subtract": ["$top", "$bot"]}}}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["AbsoluteAverageDifference"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "facet (best/worst artist + per-customer spend) + conditional averages + abs diff",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Best/worst artists by total album sales (ties by name); per-customer spend on each; absolute difference of the "
                                 "two average spends."),
        "source_schema": {"invoice_items": ["InvoiceId", "TrackId", "UnitPrice", "Quantity"], "invoices": ["InvoiceId", "CustomerId"],
                          "tracks": ["TrackId", "AlbumId"], "albums": ["AlbumId", "ArtistId"], "artists": ["ArtistId", "Name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_ii, m_inv, m_trk, m_alb, m_art],
                    "_id": "default ObjectId",
                    "indexes": ["invoices.InvoiceId", "tracks.TrackId", "albums.AlbumId", "artists.ArtistId"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Best/worst artists are chosen by total sales and ties are broken alphabetically by artist name.",
            "A customer's spend on an artist aggregates all their invoices for that artist's albums; the difference matches gold _b (4.1433).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
