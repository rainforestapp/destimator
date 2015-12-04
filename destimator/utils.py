from __future__ import unicode_literals

import pip
import subprocess


def get_current_vcs_hash():
    """Return the current git hash, if it can be found, otherwise an empty string."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except (OSError, subprocess.CalledProcessError):
        return ''


def get_installed_packages():
    return sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()])
