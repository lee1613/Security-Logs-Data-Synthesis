import numpy as np
import pandas as pd
import networkx as nx
import pytest
import src.walker as W


def _fit(rows):
    return pd.DataFrame(rows, columns=[
        "time", "src_user", "dst_user", "src_computer", "dst_computer",
        "auth_type", "logon_type", "auth_orientation", "success", "campaign_id"])


def test_fit_fanout_distributions():
    # campaign 0: foothold C1 -> C2,C3,C4 (breadth 3), users U1,U2 (2 creds), dts 60,60
    # campaign 1: foothold C1 -> C3 (breadth 1), user U1 (1 cred), no dt
    fit = _fit([
        [0,   "U1@D", "U1@D", "C1", "C2", "NTLM", "Network", "LogOn", "Success", 0],
        [60,  "U2@D", "U2@D", "C1", "C3", "NTLM", "Network", "LogOn", "Success", 0],
        [120, "U1@D", "U1@D", "C1", "C4", "NTLM", "Network", "LogOn", "Success", 0],
        [100000, "U1@D", "U1@D", "C1", "C3", "NTLM", "Network", "LogOn", "Success", 1],
    ])
    d = W.fit_fanout_distributions(fit)
    assert sorted(d["breadth"]) == [1, 3]
    assert sorted(d["creds_per_campaign"]) == [1, 2]
    assert sorted(d["inter_event_dt"]) == [60, 60]
    assert all(x > 0 for x in d["inter_event_dt"])


def test_build_host_users():
    uh = pd.DataFrame(
        [["U1@D", "C1", 5], ["U2@D", "C1", 2], ["U1@D", "C2", 1]],
        columns=["user", "computer", "cnt"])
    m = W.build_host_users(uh)
    assert m["C1"] == {"U1@D", "U2@D"}
    assert m["C2"] == {"U1@D"}
