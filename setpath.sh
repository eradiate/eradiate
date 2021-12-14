BUILD_DIR="build"

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

export PYTHONPATH="${ERADIATE_DIR}/${BUILD_DIR}/ext/mitsuba2/python:${PYTHONPATH}"
export PATH="${ERADIATE_DIR}/${BUILD_DIR}/ext/mitsuba2:${PATH}"
