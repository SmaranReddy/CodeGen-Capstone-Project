"""Test-session settings.

Tests import code out of built bundles (e.g. kaggle/spider_sql_v1_dev_bundle_v1/code/official_eval). Without this,
CPython writes __pycache__/*.pyc next to those modules, and the bundle directory then no longer matches its manifest.
"""
import sys

sys.dont_write_bytecode = True
