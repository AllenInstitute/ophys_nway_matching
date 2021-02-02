import os
import pg8000


class CredentialsException(Exception):
    pass


def get_db_credentials(
        env_prefix="LIMS_", host='limsdb2',
        database='lims2', port=5432) -> dict:
    """Get DB credentials from environment variables and keyword
    args. Environment variables must be in the format `{env_prefix}{key}`,
    where `key` is one of the following (case-sensitive):
    ['USER', 'PASSWORD', 'HOST', 'PORT', 'DATABASE'].
    Examples: LIMS_USER, LIMS_PASSWORD.
    Values for `user` and `password` must be environment variables.
    Values for `host`, `port`, and `database` can be either environment
    variables or keyword arguments, with environment variables taking
    precedence.

    Parameters
    ----------
    env_prefix : str
        expected environment variable prefix for credential keys.
    host : str
        default host value, if not found in os.environ
    database : str
        default database value, if not found in os.environ
    port : int
        default port value, if not found in os.environ

    Returns
    -------
    dict
        A dictionary of DB credentials. Contains the following fields:
        'user', 'host', 'database', 'password', 'port'.

    Raises
    ------
    CredentialsException
        Raised if user and password are not set as ENV variables
    """

    db_credentials = {}
    for key in ['user', 'password']:
        env_key = env_prefix + key.upper()
        try:
            db_credentials[key] = os.environ[env_key]
        except KeyError:
            raise CredentialsException(f"ENV variable {env_key} must be set")

    defaults = {'host': host, 'port': port, 'database': database}

    for key in ['host', 'port', 'database']:
        env_key = env_prefix + key.upper()
        db_credentials[key] = os.environ.get(env_key, defaults[key])
        if db_credentials[key] is None:
            raise CredentialsException(
                    "no ENV variable found and no default provided "
                    f"for {env_key}")

    return db_credentials


class DbConnection():

    def __init__(self, user, host, database, password, port):
        self.user = user
        self.host = host
        self.database = database
        self.password = password
        self.port = port

    @staticmethod
    def _connect(user, host, database, password, port):
        conn = pg8000.connect(user=user, host=host, database=database,
                              password=password, port=port)
        return conn, conn.cursor()

    @staticmethod
    def _select(cursor, query):
        cursor.execute(query)
        try:
            columns = [d[0].decode("utf-8") for d in cursor.description]
        except AttributeError:
            # Version 1.16.6 pg8000 values are already decoded into str
            columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, c)) for c in cursor.fetchall()]

    def query(self, query):
        conn, cursor = DbConnection._connect(self.user, self.host,
                                             self.database,
                                             self.password, self.port)

        # Guard against non-ascii characters in query
        query = ''.join([i if ord(i) < 128 else ' ' for i in query])

        try:
            results = DbConnection._select(cursor, query)
        finally:
            cursor.close()
            conn.close()
        return results
