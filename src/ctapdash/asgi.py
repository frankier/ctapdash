from . import config
from .webapp import create_app


app = create_app()


def create_app_from_env():
    """Application factory for uvicorn's reloader.

    The reloader imports the application in a spawned subprocess, so it cannot
    receive this process's parsed arguments. Configuration and the debug flag
    travel through the environment instead.
    """
    config.load_from_env()
    return create_app(debug=config.debug_from_env())
