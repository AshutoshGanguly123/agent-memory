"""Create a named tmux session with an explicit working directory and log."""
import argparse,shlex,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('mode',choices=['setup','checks','preflight','full','analysis']);a=p.parse_args()
log=root/'results'/('operations' if a.mode in ['full','analysis'] else 'setup')/(a.mode+'_launch.log')
log.parent.mkdir(parents=True,exist_ok=True)
if a.mode in ['setup','checks']:
    command='bash '+shlex.quote(str(root/'scripts'/('setup_env.sh' if a.mode=='setup' else 'checks.sh')))
elif a.mode=='analysis':
    command='source '+shlex.quote(str(root/'scripts/runtime.sh'))+'; python -m amem_reproduction.analyze --root '+shlex.quote(str(root))
else:
    command='source '+shlex.quote(str(root/'scripts/runtime.sh'))+'; python '+shlex.quote(str(root/'scripts/launch.py'))+' '+a.mode
command+=' >> '+shlex.quote(str(log))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-c',str(root),'-s','amem_hpc_'+a.mode,'bash','-c',command],check=True)
print('Started tmux amem_hpc_'+a.mode+'; log: '+str(log))
