from . import config
from .desktop import SESSION
from .webapp import create_app

app = create_app(session=SESSION)


def create_app_from_env():
    """Application factory for uvicorn's reloader.

    The reloader imports the application in a spawned subprocess, so it cannot
    receive this process's parsed arguments. Configuration and the debug flag
    travel through the environment instead. The runtime session is imported
    here rather than passed in, so the window the shared runner opens reaches
    the setup UI in the normal process too.
    """
    config.load_startup()
    return create_app(debug=config.debug_from_env(), session=SESSION)
