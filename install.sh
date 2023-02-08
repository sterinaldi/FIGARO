while getopts 'e' OPTION; do
  case "$OPTION" in
    e)
      $env_flag=1
  esac
done

if $env_flag; then
    conda env create -f figaro_env.yml
    conda activate figaro_env
fi

{python setup.py install} || {python setup.py install --user}
{python setup.py build_ext} || {python setup.py build_ext --user}

