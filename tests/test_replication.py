import os
import subprocess
from pathlib import Path
import pytest


@pytest.mark.parametrize('task,expected',[(0,'seed=43 arm=1'),(1,'seed=43 arm=3'),(2,'seed=43 arm=4'),(3,'seed=44 arm=1'),(4,'seed=44 arm=3'),(5,'seed=44 arm=4')])
def test_replication_dispatch(task,expected):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,SLURM_SUBMIT_DIR=str(root),SLURM_ARRAY_TASK_ID=str(task),SLURM_ARRAY_JOB_ID='test',SLICE_RUN='/tmp/pilot',REPLICATION_DRY_RUN='1')
    result=subprocess.run(['bash','scripts/replicate_windows.slurm'],cwd=root,env=env,text=True,capture_output=True,check=True)
    assert expected in result.stdout
    assert 'root=/tmp/pilot/replication-test' in result.stdout
