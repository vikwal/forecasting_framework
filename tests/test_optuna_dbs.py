"""optuna_dbs: URL handling of the dashboard database picker (no database needed)."""

import optuna_dbs

URL = 'postgresql://user:secret@host:5432/optuna_db'


def test_with_database_keeps_server_and_credentials():
    assert optuna_dbs.with_database(URL, 'optuna_fl') == 'postgresql://user:secret@host:5432/optuna_fl'
    assert optuna_dbs.database_name(URL) == 'optuna_db'


def test_non_postgres_url_is_its_own_only_database():
    assert optuna_dbs.list_optuna_databases('sqlite:///studies/optuna_studies.db') == ['studies/optuna_studies.db']


def test_unreachable_server_falls_back_to_own_database():
    url = 'postgresql://user:secret@127.0.0.1:1/optuna_db'
    assert optuna_dbs.list_optuna_databases(url) == ['optuna_db']
    assert not optuna_dbs.is_optuna_db(url)
