import os
import configparser
from sqlalchemy import create_engine
import pandas as pd

def get_config(config_file_path):
    """
    Lee un archivo de configuración y devuelve un objeto ConfigParser.
    """
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config

def get_database_connection(config):
    """
    Crea una conexión a la base de datos usando la configuración proporcionada.
    """
    if 'database' not in config:
        raise ValueError("La sección [database] no se encuentra en el archivo de configuración.")
    
    db_user = config['database']['user']
    db_password = config['database']['password']
    db_host = config['database']['host']
    db_name = config['database']['database']
    
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')
    return engine

def read_sql_file(file_path):
    """
    Lee el contenido de un archivo .sql y devuelve la consulta como una cadena.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def fetch_query_as_dataframe(engine, sql_file_path):
    """
    Ejecuta una consulta SQL desde un archivo .sql y devuelve un DataFrame.
    """
    query = read_sql_file(sql_file_path)
    return pd.read_sql(query, engine)