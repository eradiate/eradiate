#!/bin/bash

# Option parsing https://stackoverflow.com/a/29754866
POSITIONAL=()
OPT_ACTIVATE="NO"
OPT_DEV="NO"
OPT_ENVRC="NO"
OPT_HELP="NO"
OPT_JUPYTER="NO"
OPT_UPDATE="NO"

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
  -a|--activate)
    OPT_ACTIVATE="YES"
    shift # past argument
    ;;
  -d|--dev)
    OPT_DEV="YES"
    shift # past argument
    ;;
  -e|--envrc)
    OPT_ENVRC="YES"
    shift # past argument
    ;;
  -h|--help)
    OPT_HELP="YES"
    shift
    ;;
  -j|--jupyter)
    OPT_JUPYTER="YES"
    shift # past argument
    ;;
  -u|--update)
    OPT_UPDATE="YES"
    shift # past argument
    ;;
  *) # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift              # past argument
    ;;
  esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters
# End of option parsing

if [ ${OPT_HELP} = "YES" ]; then
  cat << EOF
usage: bash conda_create_env.sh [-a|--activate][-d|--dev][-e|--envrc][-h|--help][-j|--jupyter][-u|--update]

warning: must be executed from the eradiate root directory

options:
  -a|--activate
      Add environment variable setup to Conda environment activate script

  -d|--dev
      Install development dependencies

  -e|--envrc
      Create .envrc file for direnv-based activation

  -h|--help
      Display this help text

  -j|--jupyter
      Install Jupyter lab

  -u|--update
      Only update envs (do not overwrite)

examples:
  * default user setup:
      bash resources/envs/conda_create_env.sh -j
  * default developer setup:
      bash resources/envs/conda_create_env.sh -d -j
  * full developer setup with bells and whistles:
      bash resources/envs/conda_create_env.sh -d -j -a -e
  * developer setup update:
      bash resources/envs/conda_create_env.sh -u -d
EOF
  exit 0
fi

# Basic conda setup
eval "$(conda shell.bash hook)"
ROOT=$(pwd)
CONDA_ENV_NAME="eradiate"

if [ ${OPT_UPDATE} = "NO" ]; then
echo "Creating conda env ${CONDA_ENV_NAME} ..."
conda env create --force --quiet --file ${ROOT}/resources/deps/requirements_conda.yml --name ${CONDA_ENV_NAME} || { echo "${CONDA_ENV_NAME} env creation failed" ; exit 1 ; }
else
echo "Updating conda env ${CONDA_ENV_NAME} ..."
conda env update --quiet --file ${ROOT}/resources/deps/requirements_conda.yml --name ${CONDA_ENV_NAME} || { echo "${CONDA_ENV_NAME} env update failed" ; exit 1 ; }
fi

if [ ${OPT_DEV} = "YES" ]; then
echo "Updating conda env ${CONDA_ENV_NAME} with dev packages ..."
conda env update --quiet --file ${ROOT}/resources/deps/requirements_dev_conda.yml --name ${CONDA_ENV_NAME} || { echo "${CONDA_ENV_NAME} env update failed" ; exit 1 ; }
fi

if [ ${OPT_JUPYTER} = "YES" ]; then
echo "Updating conda env ${CONDA_ENV_NAME} with jupyter lab ..."
conda env update --quiet --file ${ROOT}/resources/deps/requirements_jupyter_conda.yml --name ${CONDA_ENV_NAME} || { echo "${CONDA_ENV_NAME} env update failed" ; exit 1 ; }
fi

conda activate ${CONDA_ENV_NAME}

# Add env activation scripts
if [ ${OPT_ACTIVATE} = "YES" ]; then
echo "Copying environment variable setup scripts ..."

mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
cat > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh << EOF
#!/bin/sh
ERADIATE_DIR="${ROOT}"
source \${ERADIATE_DIR}/setpath.sh
EOF

mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
cat > ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh << EOF
#!/bin/sh
# Clean up PATH
match="\${ERADIATE_DIR}/build/dist:"
export PATH=\${PATH//\$match/}
# Clean up PYTHONPATH
match="\${ERADIATE_DIR}/build/dist/python:"
export PYTHONPATH="\${PYTHONPATH//\$match/}"
# Remove other environment variables
unset ERADIATE_DIR MITSUBA_DIR
EOF
fi

# Create .envrc file
if [ ${OPT_ENVRC} = "YES" ]; then
echo "Creating ${ROOT}/.envrc ..."
cat > "${ROOT}/.envrc" << EOF
eval "\$(conda shell.bash hook)"
conda activate eradiate
EOF

if [ ${OPT_ACTIVATE} = "NO" ]; then
echo "source setpath.sh" >> "${ROOT}/.envrc"
fi

fi

# Enable ipywidgets support
if [ ${OPT_JUPYTER} = "YES" ]; then
echo "Enabling ipywidgets within jupyter lab ..."
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
fi

# Install Eradiate in dev mode
echo "Installing Eradiate to conda env ${CONDA_ENV_NAME} in editable mode ..."
pip install -e ${ROOT}

conda deactivate

echo
echo "Activate conda env ${CONDA_ENV_NAME} using the following command:"
echo
echo "    conda activate ${CONDA_ENV_NAME}"
echo
