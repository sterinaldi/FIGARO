env_flag=false
if [ "$1" = "-e" ]; then
  env_flag=true
fi

# If environment:
if $env_flag; then
    if ! ( conda env list | grep ".*figaro_env.*" >/dev/null 2>&1); then
        conda env create -f figaro_env.yml
    fi
    conda activate figaro_env
else
    { pip install -r requirements.txt } || { pip install --user -r requirements.txt }
fi

conda install -c conda-forge -y -S lalsuite
{ python setup.py install } || { python setup.py install --user }

