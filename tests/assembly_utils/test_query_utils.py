import nway.assembly_utils.query_utils as qu
import pytest
import os

from unittest.mock import MagicMock


@pytest.mark.parametrize(
        "env_prefix, user, password, defaults, expected",
        [
            (
                "PYTEST_DB_CREDS",
                "secret_user",
                "secret_password",
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234
                    },
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234,
                    'user': "secret_user",
                    'password': "secret_password"
                    })])
def test_get_db_credentials(
        env_prefix, user, password, defaults, expected):
    os.environ[env_prefix + 'USER'] = user
    os.environ[env_prefix + 'PASSWORD'] = password
    try:
        creds = qu.get_db_credentials(env_prefix=env_prefix, **defaults)
        assert creds == expected
    finally:
        os.environ.pop(env_prefix+'USER')
        os.environ.pop(env_prefix+'PASSWORD')


@pytest.mark.parametrize(
        "env_prefix, user, password, defaults, expected",
        [
            (
                "PYTEST_DB_CREDS",
                "secret_user",
                "secret_password",
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234,
                    },
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234,
                    'user': "secret_user",
                    'password': "secret_password"
                    }),

                ])
def test_get_db_credentials_exceptions(
        env_prefix, user, password, defaults, expected):
    with pytest.raises(qu.CredentialsException):
        qu.get_db_credentials(env_prefix=env_prefix, **defaults)


@pytest.mark.parametrize(
    "description",
    [
        [(b"col1", 23, None), (b"col2", 1043, None)],
        [("col1", 23, None), ("col2", 1043, None)],
    ], ids=["pg8000_v1.16.5", "pg8000_v1.16.6+"]
)
def test_decoding_columns(description):
    """Test that the changes in encoding from pg8000 1.16.5 to 1.16.6
    are handled appropriately.
    """
    expected = [{"col1": 1997, "col2": "totale"},
                {"col1": 2000, "col2": "finsternis"}]
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_cursor.description = description
    mock_cursor.fetchall.return_value = (
        [1997, 'totale'], [2000, 'finsternis'])
    actual = qu.DbConnection._select(mock_cursor, "SELECT...")
    assert expected == actual
