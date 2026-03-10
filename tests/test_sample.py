# tests/test_sample.py

def test_math_basique():
    # Un test trivial pour vérifier que pytest s'exécute correctement
    assert 2 + 2 == 4


def test_string_contains():
    message = "coding week"
    assert "coding" in message