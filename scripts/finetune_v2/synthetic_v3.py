"""Synthetic diagnosis for verifier-3: hand-built SQLite databases shaped like the escape classes
reported (at class level only) by the iteration-2 holdout: multi-table joins (inner->left,
lookup field swap, != / NULL guards) and ORDER BY ... LIMIT 1 (dropped sort). No Spider record
is used. For every query: convert with sql2mql-v2, verify with verifier-2 AND verifier-3, then
every single-edit mutant (mutants_v2) through both verifiers; verifier-accepted mutants are
judged by the oracle battery (oracle_v2) exactly as in the mutant study.

    python -B scripts/finetune_v2/synthetic_v3.py   -> outputs/sql2mql_v2/synthetic_v3/
"""
import json
import sqlite3
from collections import Counter

import v2_common as C
import mutants_v2 as MU
import oracle_v2 as O
import schema_profile as SP
import sql2mql_v2 as M
import verifier as V2
import verifier3 as V3

OUT = C.OUT / "synthetic_v3"
DBS = OUT / "dbs"

SCHEMAS = {
    # club / membership / student chain; nullable FKs; data inserted in key order
    "syn_chain": """
        CREATE TABLE club (clubid INTEGER PRIMARY KEY, clubname TEXT, clubloc TEXT);
        CREATE TABLE member_of_club (stuid INTEGER, clubid INTEGER, position TEXT);
        CREATE TABLE student (stuid INTEGER PRIMARY KEY, lname TEXT, fname TEXT, age INTEGER,
                              advisor INTEGER, city_code TEXT);
        INSERT INTO club VALUES (1,'Bootup Baltimore','AKW'),(2,'Hopkins Student Enterprises','AKW'),
                                (3,'Pen and Paper Gaming','HHH'),(4,'Tennis Club','AKW');
        INSERT INTO student VALUES (1001,'Smith','Linda',18,1121,'BAL'),(1002,'Kim','Tracy',19,7712,'HKG'),
            (1003,'Jones','Shiela',21,7792,'WAS'),(1004,'Kumar','Dinesh',20,8423,'CHI'),
            (1005,'Gompers','Paul',26,1121,'YYZ'),(1006,'Schultz','Andy',18,1148,'BAL'),
            (1007,'Apap','Lisa',18,8918,'PIT'),(1008,'Nelson','Jandy',20,9172,'BAL');
        INSERT INTO member_of_club VALUES (1001,1,'President'),(1007,4,'Vice-President'),
            (1008,2,'CTO'),(1010,1,'Teaching Coordinator'),(1002,3,NULL),(NULL,2,'Member');
    """,
    # orders / products / customers with an ORDER BY ... LIMIT 1 over a join; ids coincide 1..n
    "syn_shop": """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, pname TEXT, price REAL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
                             qty INTEGER, status TEXT);
        INSERT INTO customers VALUES (1,'Ann','Rome'),(2,'Bob','Oslo'),(3,'Cy','Rome'),(4,'Di',NULL);
        INSERT INTO products VALUES (1,'pen',1.5),(2,'ink',3.0),(3,'pad',4.25),(4,'box',9.0);
        INSERT INTO orders VALUES (1,1,1,2,'paid'),(2,1,2,1,'open'),(3,2,3,5,'paid'),(4,3,4,1,NULL),
                                  (5,NULL,2,3,'paid'),(6,4,1,7,'open');
    """,
    # single table stored in ascending score order (natural order == sort order)
    "syn_rank": """
        CREATE TABLE player (pid INTEGER PRIMARY KEY, name TEXT, score INTEGER, team TEXT, age INTEGER);
        INSERT INTO player VALUES (1,'a',10,'red',20),(2,'b',20,'red',31),(3,'c',30,'blue',25),
                                  (4,'d',40,'blue',NULL),(5,'e',50,NULL,28),(6,'f',60,'red',22);
    """,
}

QUERIES = [
    ("syn_chain", "SELECT count(*) FROM club AS t1 JOIN member_of_club AS t2 ON t1.clubid = t2.clubid "
                  "JOIN student AS t3 ON t2.stuid = t3.stuid WHERE t1.clubname = 'Bootup Baltimore'"),
    ("syn_chain", "SELECT t3.fname, t3.lname FROM club AS t1 JOIN member_of_club AS t2 ON t1.clubid = t2.clubid "
                  "JOIN student AS t3 ON t2.stuid = t3.stuid WHERE t1.clubname = 'Hopkins Student Enterprises'"),
    ("syn_chain", "SELECT DISTINCT t1.clubname FROM club AS t1 JOIN member_of_club AS t2 ON t1.clubid = t2.clubid "
                  "JOIN student AS t3 ON t2.stuid = t3.stuid WHERE t3.city_code = 'BAL'"),
    ("syn_chain", "SELECT t1.clubname FROM club AS t1 JOIN member_of_club AS t2 ON t1.clubid = t2.clubid "
                  "JOIN student AS t3 ON t2.stuid = t3.stuid WHERE t2.position != 'President'"),
    ("syn_chain", "SELECT count(*) FROM club AS t1 JOIN member_of_club AS t2 ON t1.clubid = t2.clubid "
                  "JOIN student AS t3 ON t2.stuid = t3.stuid WHERE t3.age > 19 AND t1.clubloc = 'AKW'"),
    ("syn_shop", "SELECT T1.name FROM customers AS T1 JOIN orders AS T2 ON T1.id = T2.customer_id "
                 "JOIN products AS T3 ON T2.product_id = T3.id ORDER BY T3.price DESC LIMIT 1"),
    ("syn_shop", "SELECT T3.pname FROM customers AS T1 JOIN orders AS T2 ON T1.id = T2.customer_id "
                 "JOIN products AS T3 ON T2.product_id = T3.id WHERE T1.city = 'Rome' AND T2.status != 'open'"),
    ("syn_shop", "SELECT T1.name, T2.qty FROM customers AS T1 JOIN orders AS T2 ON T1.id = T2.customer_id "
                 "WHERE T2.qty > 2"),
    ("syn_shop", "SELECT count(*) FROM orders AS T2 JOIN products AS T3 ON T2.product_id = T3.id "
                 "WHERE T3.price > 2"),
    ("syn_rank", "SELECT name FROM player ORDER BY score DESC LIMIT 1"),
    ("syn_rank", "SELECT name FROM player ORDER BY score LIMIT 1"),
    ("syn_rank", "SELECT name, age FROM player WHERE team = 'red' ORDER BY score DESC LIMIT 1"),
    ("syn_rank", "SELECT team FROM player WHERE age > 21 ORDER BY age LIMIT 1"),
    ("syn_rank", "SELECT name FROM player WHERE team != 'blue' AND score >= 20"),
]


def db_path(db_id):
    return DBS / f"{db_id}.sqlite"


def build():
    DBS.mkdir(parents=True, exist_ok=True)
    for db, ddl in SCHEMAS.items():
        p = db_path(db)
        if p.exists():
            p.unlink()
        con = sqlite3.connect(p)
        con.executescript(ddl)
        con.commit()
        con.close()


def main():
    build()
    client = C.local_client()
    C.check_server(client)
    r2, r3 = V2.RealDbs(client, path_of=db_path), V3.RealDbs(client, path_of=db_path)
    out, labels = [], []
    for k, (db, sql) in enumerate(QUERIES):
        item = {"id": f"syn{k:02d}", "db_id": db, "sql": sql}
        try:
            mql, _ = M.convert(sql, SP.Profile(db, db_path(db)))
        except M.Unsupported as e:
            labels.append({"id": item["id"], "sql": sql, "converted": False, "code": e.code})
            continue
        v2o, v3o = V2.verify(item, mql, r2)["status"], V3.verify(item, mql, r3)["status"]
        lab = {"id": item["id"], "sql": sql, "converted": True, "verifier2": v2o, "verifier3": v3o}
        labels.append(lab)
        if v2o != "accepted":
            continue
        oracle, base = None, None
        for op, m in MU.mutants(mql)[:30]:
            a2, a3 = V2.verify(item, m, r2)["status"], V3.verify(item, m, r3)
            row = {"label": item["id"], "operator": op, "verifier2": a2, "verifier3": a3["status"],
                   "v3_detail": a3.get("detail")}
            if a2 == "accepted" or a3["status"] == "accepted":
                if base is None:
                    oracle, base = O.battery(item, mql, r2)
                _, res = O.battery(item, m, r2, oracle)
                sep = [f for f, ok in res.items() if base.get(f) is True and ok is not True]
                row["oracle"] = "non_equivalent" if sep else "unkilled"
                row["oracle_separating"] = sep[:5]
            out.append(row)
        print(item["id"], v2o, v3o, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(OUT / "labels.jsonl", labels)
    C.write_jsonl(OUT / "mutants.jsonl", out)
    fa2 = [x for x in out if x["verifier2"] == "accepted" and x.get("oracle") == "non_equivalent"]
    fa3 = [x for x in out if x["verifier3"] == "accepted" and x.get("oracle") == "non_equivalent"]
    summ = {
        "queries": len(QUERIES), "converted": sum(x["converted"] for x in labels),
        "label_status": {"verifier2": dict(Counter(x.get("verifier2") for x in labels if x["converted"])),
                         "verifier3": dict(Counter(x.get("verifier3") for x in labels if x["converted"]))},
        "mutants": len(out),
        "verifier2_false_accepts": [(x["label"], x["operator"]) for x in fa2],
        "verifier3_false_accepts": [(x["label"], x["operator"]) for x in fa3],
        "labels_with_false_accept": {"verifier2": len({x["label"] for x in fa2}),
                                     "verifier3": len({x["label"] for x in fa3})},
        "v3_kill_reasons_for_v2_escapes": dict(Counter(x["verifier3"] for x in fa2)),
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
