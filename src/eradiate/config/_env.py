from __future__ import annotations

import os

#: Identifier of the environment in which Eradiate is used. Takes the value of
#: the ``ERADIATE_ENV`` environment variable if it is set; otherwise defaults to
#: ``"default"``.
ENV: str = os.environ.get("ERADIATE_ENV", "default")
