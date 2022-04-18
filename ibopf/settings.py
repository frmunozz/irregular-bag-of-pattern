"""Module to parse the settings file (2nd step after avocado).

This file first reads the default settings file, and then optionally reads a
settings file in the working directory to override any of the settings.

adapted setting load based on avocado settings, in this adapted versions, a second level dictionary
is added to settings, one for avocado and one for ibopf
"""
import json
import os


def load_settings():
    # Figure out the root directory for our package.
    dirname = os.path.dirname
    package_root_directory = dirname(dirname(os.path.abspath(__file__)))

    # First, load the default settings
    default_path = os.path.join(package_root_directory, "settings.json")
    settings = json.load(open(default_path))

    # Next, override with user settings if exists
    user_path = os.path.join(os.getcwd(), settings["user_settings_file"])
    try:
        user_settings = json.load(open(user_path))
    except FileNotFoundError:
        # No user settings available. Just use the defaults.
        pass
    else:
        for k, v in user_settings.items():
            if k not in settings.keys():
                continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if k2 not in settings[k].keys():
                        continue
                    settings[k][k2] = v2
            else:
                settings[k] = v
    return settings

settings = load_settings()


