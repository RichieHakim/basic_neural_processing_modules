from typing import Optional, Union, List, Tuple, Dict, Any, Iterable, Sequence

import urllib

import sqlalchemy
import pandas as pd


def make_url(
    sql_type: str = "mysql",
    username: str = "root",
    password: str = "password",
    host: str = "localhost",
    port: int = None,
    database: str = "test",
    quote_password: bool = False,
) -> str:
    """
    Formats a URL for a SQL database connection.\n
    Works with MySQL, PostgreSQL, and SQLite.\n
    RH 2024

    Args:
        sql_type (str):
            Type of SQL database. Options are "mysql", "postgresql", or "sqlite"
        username (str):
            Username for the database
        password (str):
            Password for the database
        host (str):
            Host for the database
        port (int):
            Port for the database.\n
            If 'default', then the default port for the SQL type is used.\n
        database (str):
            Name of the database
        quote_password (bool):
            Use quote_plus to format the passwords with special characters like:
            !@#$%^&*(){}/[]|:;<>,.?\~`_+-=\n

    Returns:
        (str):
            URL for the database connection
    """
    if (port is not None) and (port != "default"):
        port = f":{port}"
    elif port is None:
        port = ""

    if quote_password:
        password = urllib.parse.quote_plus(password)

    if sql_type == "mysql":
        if port == "default":
            port = 3306
        return f"mysql+pymysql://{username}:{password}@{host}{port}/{database}"
    
    elif sql_type == "postgresql":
        if port == "default":
            port = 5432
        return f"postgresql+psycopg2://{username}:{password}@{host}{port}/{database}"
    
    elif sql_type == "sqlite":
        return f"sqlite:///{database}"
    
    else:
        raise ValueError("sql_type must be 'mysql', 'postgresql', or 'sqlite'")
    

def parse_url(
    url: str
) -> Dict[str, str]:
    """
    Parses / decomposes a URL for a SQL database connection into its components.
    \n
    RH 2024

    Args:
        url (str):
            URL for the database connection

    Returns:
        (dict):
            Dictionary of the decomposed URL
    """

    def assert_split_in_n(s: str, split: str, n=2):
        assert len(s.split(split)) == n, f"URL must have exactly {n-1} '{split}' characters"

    if "mysql" in url:
        sql_type = "mysql"
    elif "postgresql" in url:
        sql_type = "postgresql"
    elif "sqlite" in url:
        sql_type = "sqlite"
    else:
        raise ValueError("URL type not recognized")
    
    ## Find the :// or :/// and remove everything before it
    if sql_type == "mysql":
        assert_split_in_n(url, "://", n=2)
        url = url.split("://")[1]
    elif sql_type == "postgresql":
        assert_split_in_n(url, "://", n=2)
        url = url.split("://")[1]
    elif sql_type == "sqlite":
        assert_split_in_n(url, "://", n=2)
        url = url.split(":///")[1]
        
    ## Split the username and password
    username, password_host_port_database = url.split(":")[0], url.split(":")[1]

    ## Split the password, host, port, and database
    assert_split_in_n(password_host_port_database, "@", n=2)
    password, host_port_database = password_host_port_database.split("@")
    
    ## Split the host, port, and database
    ### Handle if there is no port and if there is no database
    if "/" in host_port_database:
        assert_split_in_n(host_port_database, "/", n=2)
        host_port, database = host_port_database.split("/")
    else:
        host_port, database = host_port_database, None
    
    ## Split the host and port
    if ":" in host_port:
        assert_split_in_n(host_port, ":", n=2)
        host, port = host_port.split(":")
    else:
        host, port = host_port, None

    return {
        "sql_type": sql_type,
        "username": username,
        "password": password,
        "host": host,
        "port": port,
        "database": database
    }


def make_sql_connection(
    url: str,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_recycle: int = 3600,
    pool_pre_ping: bool = True,
    **kwargs,
) -> sqlalchemy.engine.base.Connection:
    """
    Makes a connection to a SQL database.
    RH 2024

    Args:
        url (str):
            URL for the database connection
        echo (bool):
            Whether to echo SQL commands to the console
        pool_size (int):
            Number of connections to keep in the pool
        max_overflow (int):
            Maximum number of connections to create
        pool_recycle (int):
            Number of seconds after which to recycle a connection
        pool_pre_ping (bool):
            Whether to ping the server before using a connection
        **kwargs:
            Additional keyword arguments for the connection

    Returns:
        (sqlalchemy.engine.base.Connection):
            SQL connection object
    """
    return sqlalchemy.create_engine(
        url=url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_recycle=pool_recycle,
        pool_pre_ping=pool_pre_ping,
        **kwargs
    ).connect()


def get_available_databases(
    connection: sqlalchemy.engine.base.Connection,
) -> dict:
    """
    Gets a list of all databases available on the server.
    RH 2024
    
    Args:
        connection (sqlalchemy.engine.base.Connection):
            SQL connection object
    
    Returns:
        (pd.DataFrame):
            DataFrame of all databases
    """
    if isinstance(connection, sqlalchemy.engine.base.Connection):
        if "mysql" in str(connection.engine):
            query = "SHOW DATABASES"
        elif "postgresql" in str(connection.engine):
            query = "SELECT datname FROM pg_database WHERE datistemplate = false"
        elif "sqlite" in str(connection.engine):
            raise ValueError("SQLite does not support this operation")
        else:
            raise ValueError("Connection type not recognized")
    else:
        raise TypeError("connection must be a sqlalchemy.engine.base.Connection object")
    
    return pd.read_sql(query, connection)


def get_available_tables(
    connection: sqlalchemy.engine.base.Connection,
    database: Optional[str] = None,
) -> dict:
    """
    Gets a list of all tables available in the database.
    RH 2024
    
    Args:
        connection (sqlalchemy.engine.base.Connection):
            SQL connection object
        database (str):
            Name of the database. If None, then the default database is used.
    
    Returns:
        (pd.DataFrame):
            DataFrame of all tables
    """
    if isinstance(connection, sqlalchemy.engine.base.Connection):
        if "mysql" in str(connection.engine):
            if database is None:
                raise ValueError("database must be specified for MySQL")
            query = f"SHOW TABLES IN {database}"
        elif "postgresql" in str(connection.engine):
            if database is None:
                query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'"
            else:
                query = f"SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema' AND tablename NOT LIKE 'pg_%' AND tablename NOT LIKE 'sql_%'"
        elif "sqlite" in str(connection.engine):
            raise ValueError("SQLite does not support this operation")
        else:
            raise ValueError("Connection type not recognized")
    else:
        raise TypeError("connection must be a sqlalchemy.engine.base.Connection object")
    
    return pd.read_sql(query, connection)


def get_table_columns(
    connection: sqlalchemy.engine.base.Connection,
    table: str,
    database: Optional[str] = None,
) -> dict:
    """
    Gets a list of all columns in the table.
    RH 2024
    
    Args:
        connection (sqlalchemy.engine.base.Connection):
            SQL connection object
        table (str):
            Name of the table
        database (str):
            Name of the database. If None, then the default database is used.
    
    Returns:
        (pd.DataFrame):
            DataFrame of all columns
    """
    if isinstance(connection, sqlalchemy.engine.base.Connection):
        if "mysql" in str(connection.engine):
            if database is None:
                raise ValueError("database must be specified for MySQL")
            query = f"SHOW COLUMNS FROM {database}.{table}"
        elif "postgresql" in str(connection.engine):
            if database is None:
                query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
            else:
                query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' AND table_catalog = '{database}'"
        elif "sqlite" in str(connection.engine):
            raise ValueError("SQLite does not support this operation")
        else:
            raise ValueError("Connection type not recognized")
    else:
        raise TypeError("connection must be a sqlalchemy.engine.base.Connection object")
    
    return pd.read_sql(query, connection)


def get_table_data(
    connection: sqlalchemy.engine.base.Connection,
    table: str,
    database: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Gets a list of all data in the table.
    RH 2024
    
    Args:
        connection (sqlalchemy.engine.base.Connection):
            SQL connection object
        table (str):
            Name of the table
        database (str):
            Name of the database. If None, then the default database is used.
        limit (int):
            Number of rows to return. If None, then all rows are returned.
    
    Returns:
        (pd.DataFrame):
            DataFrame of all data
    """
    if isinstance(connection, sqlalchemy.engine.base.Connection):
        if "mysql" in str(connection.engine):
            if database is None:
                raise ValueError("database must be specified for MySQL")
            query = f"SELECT * FROM {database}.{table}"
        elif "postgresql" in str(connection.engine):
            if database is None:
                query = f"SELECT * FROM {table}"
            else:
                query = f"SELECT * FROM {database}.{table}"
        elif "sqlite" in str(connection.engine):
            if database is not None:
                raise ValueError("database must be None for SQLite")
            query = f"SELECT * FROM {table}"
        else:
            raise ValueError("Connection type not recognized")
    else:
        raise TypeError("connection must be a sqlalchemy.engine.base.Connection object")
    
    if limit is not None:
        query += f" LIMIT {limit}"
    
    return pd.read_sql(query, connection)


def create_database(
    connection: sqlalchemy.engine.base.Connection,
    database: str,
) -> None:
    """
    Creates a new database on the server.
    RH 2024
    
    Args:
        connection (sqlalchemy.engine.base.Connection):
            SQL connection object
        database (str):
            Name of the database to create
    """
    if isinstance(connection, sqlalchemy.engine.base.Connection):
        if "mysql" in str(connection.engine):
            query = sqlalchemy.text(f"CREATE DATABASE {database}")
        elif "postgresql" in str(connection.engine):
            query = sqlalchemy.text(f"CREATE DATABASE {database}")
        elif "sqlite" in str(connection.engine):
            raise ValueError("SQLite does not support this operation")
        else:
            raise ValueError("Connection type not recognized")
    else:
        raise TypeError("connection must be a sqlalchemy.engine.base.Connection object")
    
    connection.execute(query)


def drop_database(
    connection: sqlalchemy.engine.base.Connection,
    database: str,
) -> None:
    """
    Drops a database from the server.
    RH 2024
    
    Args:
        connection (sqlalchemy.engine.base.Connection):
            SQL connection object
        database (str):
            Name of the database to drop
    """
    if isinstance(connection, sqlalchemy.engine.base.Connection):
        if "mysql" in str(connection.engine):
            query = f"DROP DATABASE {database}"
        elif "postgresql" in str(connection.engine):
            query = f"DROP DATABASE {database}"
        elif "sqlite" in str(connection.engine):
            raise ValueError("SQLite does not support this operation")
        else:
            raise ValueError("Connection type not recognized")
    else:
        raise TypeError("connection must be a sqlalchemy.engine.base.Connection object")
    
    connection.execute(sqlalchemy.text(query))


def drop_table(
    connection: sqlalchemy.engine.base.Connection,
    table: str,
    database: Optional[str] = None,
) -> None:
    """
    Drops a table from the database.
    RH 2024
    
    Args:
        connection (sqlalchemy.engine.base.Connection):
            SQL connection object
        table (str):
            Name of the table to drop
        database (str):
            Name of the database. If None, then the default database is used.
    """
    if isinstance(connection, sqlalchemy.engine.base.Connection):
        if "mysql" in str(connection.engine):
            if database is None:
                raise ValueError("database must be specified for MySQL")
            query = f"DROP TABLE {database}.{table}"
        elif "postgresql" in str(connection.engine):
            if database is None:
                query = f"DROP TABLE {table}"
            else:
                query = f"DROP TABLE {database}.{table}"
        elif "sqlite" in str(connection.engine):
            if database is not None:
                raise ValueError("database must be None for SQLite")
            query = f"DROP TABLE {table}"
        else:
            raise ValueError("Connection type not recognized")
    else:
        raise TypeError("connection must be a sqlalchemy.engine.base.Connection object")
    
    connection.execute(sqlalchemy.text(query))