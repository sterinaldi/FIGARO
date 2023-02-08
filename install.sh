env_flag=false
if [ "$1" = "-e" ]; then
  env_flag=true
fi

if $env_flag; then
    if ! ( conda env list | grep ".*figaro_env.*" >/dev/null 2>&1); then
        conda env create -f figaro_env.yml
    fi
    conda activate figaro_env
fi

{python setup.py install} || {python setup.py install --user}

