import gzip
import pytest

# 9-field auth: time,src_user,dst_user,src_computer,dst_computer,
#               auth_type,logon_type,auth_orientation,success
MINI_AUTH = (
    "1,U1@D1,U1@D1,C1,C2,Ntlm,Network,LogOn,Success\n"
    "2,U1@D1,U1@D1,C1,C2,Ntlm,Network,LogOn,Success\n"
    "3,U2@D1,U2@D1,C2,C3,Kerberos,Interactive,LogOn,Success\n"
    "4,U3@D1,U3@D1,C1,C3,Ntlm,Network,LogOn,Failure\n"
    "3601,U2@D1,U2@D1,C3,C4,?,Network,LogOff,Success\n"
    "7201,U4@D1,U4@D1,C4,C3,Kerberos,Network,LogOn,Success\n"
)

# 4-field redteam: time,user,src_computer,dst_computer
# First two rows exist in MINI_AUTH (join); third does not (unmatched).
MINI_REDTEAM = (
    "3,U2@D1,C2,C3\n"
    "7201,U4@D1,C4,C3\n"
    "99999,U9@D1,C9,C9\n"
)


def _write_gz(path, text):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)
    return path


@pytest.fixture
def mini_auth_path(tmp_path):
    return _write_gz(tmp_path / "mini_auth.txt.gz", MINI_AUTH)


@pytest.fixture
def mini_redteam_path(tmp_path):
    return _write_gz(tmp_path / "mini_redteam.txt.gz", MINI_REDTEAM)
