
export PYTHONPATH=$PYTHONPATH:"/home/adas/python/"
export PYTHONPATH=$PYTHONPATH:"home/jhl7340/"
export PYTHONPATH=$PYTHONPATH:"home/jhl7340/PESDT/"
export PYTHONPATH=$PYTHONPATH:"home/jhl7340/eproc/EPROC/python/eproc"
export PYTHONPATH=$PYTHONPATH:"home/jhl7340/eproc/EPROC/python"
export CHERAB_CADMESH='/common/cadmesh/'

module unload python/2.7.5
module load python/3.9
module load jet
a=var$(python3 --version)
echo $a


