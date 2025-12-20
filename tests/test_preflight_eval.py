from __future__ import annotations

from scripts.preflight_eval import parse_tasklist_csv


def test_parse_tasklist_csv_extracts_python_pids_and_mem():
    sample = (
        '"Image Name","PID","Session Name","Session#","Mem Usage"\n'
        '"python.exe","1234","Console","1","5,888 K"\n'
        '"Python.exe","9999","Console","1","12,345 K"\n'
        '"notepad.exe","222","Console","1","1,000 K"\n'
    )
    procs = parse_tasklist_csv(sample)
    assert [p.pid for p in procs] == [1234, 9999]
    assert [p.mem_kb for p in procs] == [5888, 12345]


def test_parse_tasklist_csv_empty_ok():
    assert parse_tasklist_csv("") == []

