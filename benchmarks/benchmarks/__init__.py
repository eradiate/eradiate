import sys
import os

if "ASV" in os.environ:
    asv_env_dir = os.environ["ASV_ENV_DIR"]

    os.environ["ERADIATE_SOURCE_DIR"] = os.environ["ASV_ENV_DIR"]+"/project"
    eradiate_source_dir = os.environ["ERADIATE_SOURCE_DIR"]

    sys.path.insert(0, eradiate_source_dir+"/ext/mitsuba/build/python")

    print("INIT SYS PATH:")
    print(sys.path)
