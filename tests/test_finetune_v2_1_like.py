"""sql2mql-v2.1 LIKE translation, unit level (no MongoDB). Plan §B.4 test 1.

Non-ASCII test characters are written ONLY as \\u escapes (U+212A has a canonical decomposition to ASCII
K, so a pasted glyph can silently become ASCII under NFC normalisation).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
import schema_profile as SP  # noqa: E402
import sql2mql_v2 as V2  # noqa: E402
import sql2mql_v2_1 as V21  # noqa: E402

KELVIN, LONG_S = "K", "ſ"
CIRCLED_A, CIRCLED_a = "Ⓐ", "ⓐ"
ROMAN_I, ROMAN_i = "Ⅰ", "ⅰ"


def test_escapes_are_not_ascii():
    assert ord(KELVIN) == 0x212A and ord(LONG_S) == 0x17F and KELVIN != "K"
    assert not CIRCLED_A.isalpha() and not ROMAN_i.isalpha()   # why v2's guard let them through


@pytest.mark.parametrize("pattern,rx", [
    ("%Scanner%", "[sS][cC][aA][nN][nN][eE][rR]"),
    ("%6862 Kaitlyn Knolls%", "6862 [kK][aA][iI][tT][lL][yY][nN] [kK][nN][oO][lL][lL][sS]"),
    ("a_b", "^[aA].[bB]\\z"),
    ("abc%", "^[aA][bB][cC]"),
    ("%abc", "[aA][bB][cC]\\z"),
    ("a%b", "^[aA].*[bB]\\z"),
    ("", "^\\z"),
    ("%", "\\z"),
    ("%%", ""),
    ("12-34", "^12-34\\z"),
    ("% x %", " [xX] "),
    ("_", "^.\\z"),
    ("%.^$|?*+()[]{}/\\%", "\\.\\^\\$\\|\\?\\*\\+\\(\\)\\[\\]\\{\\}\\/\\\\"),
    ("%" + CIRCLED_A + "%", CIRCLED_A),
    (ROMAN_i, "^" + ROMAN_i + "\\z"),
])
def test_exact_regex(pattern, rx):
    assert V21.like_regex_ascii(pattern) == rx


def test_same_anchors_and_wildcards_as_v2():
    """v2.1 = v2's regex with ASCII letters replaced by classes, for patterns without a backslash."""
    for p in ["%Scanner%", "a_b", "abc%", "%abc", "a%b", "", "%", "%%", "_%_", "%_", "x.y%", "%(a)%"]:
        v2 = V2.like_regex(p)
        expanded = "".join("[" + c.lower() + c.upper() + "]" if c.isascii() and c.isalpha() else c for c in v2)
        # v2's escapes are '\\' + char; an escaped letter does not occur (no letter is in REGEX_META), and
        # '\\z' is an anchor: undo its expansion
        expanded = expanded.replace("\\[zZ]", "\\z")
        assert V21.like_regex_ascii(p) == expanded, p


@pytest.fixture(scope="module")
def prof():
    return SP.Profile("store_product")


def _pipe(sql, prof, mod=V21):
    return json.dumps(mod.convert(sql, prof)[0]["pipeline"])


def test_options_s_in_all_three_paths(prof):
    pos = _pipe('SELECT product FROM product WHERE product LIKE "%Scanner%"', prof)
    neg = _pipe('SELECT product FROM product WHERE product NOT LIKE "%Scanner%"', prof)
    hav = _pipe('SELECT product, count(*) FROM product GROUP BY product HAVING product LIKE "%S%"', prof)
    assert '"$options": "s"' in pos and '"$regex": "[sS][cC][aA][nN][nN][eE][rR]"' in pos
    assert '"$not": {"$regex"' in neg and '"$options": "s"' in neg and '"$ne": null' in neg
    assert '"$regexMatch"' in hav and '"options": "s"' in hav and '"$type"' in hav
    for p in (pos, neg, hav):
        assert '"is"' not in p and '"i"' not in p


def test_v2_still_emits_is(prof):
    assert '"$options": "is"' in _pipe('SELECT product FROM product WHERE product LIKE "%Scanner%"', prof, V2)


@pytest.mark.parametrize("sql,code", [
    ('SELECT product FROM product WHERE product LIKE "%é%"', "like"),       # non-ASCII letter
    ('SELECT product FROM product WHERE dpi LIKE "%1%"', "like"),               # LIKE on num column
    ('SELECT product FROM product WHERE interface LIKE dimensions', "like"),    # not a literal
])
def test_rejections_unchanged(prof, sql, code):
    for mod in (V2, V21):
        with pytest.raises(mod.Unsupported) as e:
            mod.convert(sql, prof)
        assert e.value.code == code


@pytest.mark.parametrize("ch", [CIRCLED_A, ROMAN_i])
def test_case_paired_nonletters_accepted_and_translated_without_i(prof, ch):
    sql = f'SELECT product FROM product WHERE product LIKE "%{ch}%"'
    m2 = V2.convert(sql, prof)[0]
    m21 = V21.convert(sql, prof)[0]
    assert m2["pipeline"][0]["$match"]["product"] == {"$regex": ch, "$options": "is"}
    assert m21["pipeline"][0]["$match"]["product"] == {"$regex": ch, "$options": "s"}


def test_non_like_identical_to_v2(prof):
    for sql in ['SELECT product FROM product WHERE dpi > 600 ORDER BY product',
                'SELECT count(*) FROM product', 'SELECT max(dpi), interface FROM product GROUP BY interface',
                'SELECT T1.Store_Name FROM store AS T1 JOIN store_product AS T2 ON T1.Store_ID = T2.Store_ID']:
        assert V2.convert(sql, prof) == V21.convert(sql, prof)


def test_version_and_independence():
    assert V21.CONVERTER_VERSION == "sql2mql-v2.1" and V21.BASE_CONVERTER_VERSION == "sql2mql-v2"
    src = (ROOT / "scripts/finetune_v2/sql2mql_v2_1.py").read_text(encoding="utf-8")
    for bad in ("import verifier", "import fixtures", "import mongo_import", "import comparator"):
        assert bad not in src
