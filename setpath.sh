if [[ "$#" -ge "1" ]]; then
    BUILD_DIR="$1"
else
    BUILD_DIR="build"
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "The setpath.sh script must be sourced, not executed. In other words, run\n"
    echo "$ source setpath.sh\n"
    exit 0
fi

if [ "$BASH_VERSION" ]; then
    ERADIATE_DIR=$(dirname "$BASH_SOURCE")
    export ERADIATE_DIR=$(builtin cd "$ERADIATE_DIR"; builtin pwd)
elif [ "$ZSH_VERSION" ]; then
    export ERADIATE_DIR=$(dirname "$0:A")
fi

export PYTHONPATH="$ERADIATE_DIR/dist/python:$ERADIATE_DIR/$BUILD_DIR/dist/python:$PYTHONPATH"
export PATH="$ERADIATE_DIR/dist:$ERADIATE_DIR/$BUILD_DIR/dist:$PATH"
export MITSUBA_DIR="$ERADIATE_DIR"/ext/mitsuba2