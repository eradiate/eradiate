conda create --name eradiate -y
conda init bash
. /opt/conda/etc/profile.d/conda.sh
conda activate eradiate
make conda-init
make kernel