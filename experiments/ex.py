from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

# Try to locate config file for Mongo DB
import importlib
spec = importlib.util.find_spec('mongodburi')
if spec is not None:
    from mongodburi import mongo_uri, db_name
else:
    mongo_uri, db_name = None, None

from incense import ExperimentLoader
def get_loader(uri=mongo_uri, db=db_name):
    loader = ExperimentLoader(
        mongo_uri=uri,
        db_name=db
    )
    return loader


def get_logger(_run):
    def log_run(key, val):
        _run.info[key] = val
    from experiments.logger import log
    if mongo_uri is not None and db_name is not None:
        # log = _run.log_scalar # for numerical series
        log = log_run

    return log


def new_ex(interactive=True):
    ex = Experiment('jupyter_ex', interactive=interactive)
    # ex.captured_out_filter = apply_backspaces_and_linefeeds
    ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
    if mongo_uri is not None and db_name is not None:
        ex.observers.append(MongoObserver(url=mongo_uri, db_name=db_name))

    return ex
