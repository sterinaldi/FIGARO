env_flag=false
if [ "$1" = "-e" ]; then
  env_flag=true
fi

# If environment:
if $env_flag; then
    if ! ( conda env list | grep ".*figaro_env.*" >/dev/null 2>&1); then
        conda env create -f figaro_env.yml
    fi
    if [[ $CONDA_DEFAULT_ENV!='figaro_env' ]]; then
        conda activate figaro_env
    fi

# Install lalsuite from CONDA
conda install -c conda-forge -y -S lalsuite

{
pip install -r requirements.txt
} || {
pip install --user -r requirements.txt
}

python setup.py build_ext --inplace

{
python setup.py install
} || {
python setup.py install --user
}

