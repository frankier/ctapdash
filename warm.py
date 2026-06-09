import click
from os import environ
import tomlkit
from pathlib import Path
from ctapdash.io import read_eeglab


@click.command()
@click.option("--clean", is_flag=True)
def main(clean):
    from ctapdash.io import cached_path
    with open(environ["CTAPDASH_SETTINGS"]) as f:
        settings = tomlkit.parse(f.read())
    sources_settings = settings["sources"]
    for source in sources_settings:
        source_path = Path(sources_settings[source])
        for subdir in source_path.iterdir():
            if not subdir.name[0].isnumeric():
                continue
            for path in subdir.iterdir():
                if path.suffix == ".set":
                    if clean:
                        cached_path(path, raw=True).unlink(missing_ok=True)
                        cached_path(path, raw=False).unlink(missing_ok=True)
                    if read_eeglab(path, warm=True):
                        print("Caching:", path)
                    else:
                        print("Already cached:", path)


if __name__ == "__main__":
    main()
