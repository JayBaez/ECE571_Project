import argparse, subprocess, sys

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='data.xlsx'); p.add_argument('--fast',action='store_true'); a=p.parse_args(); flag=['--fast'] if a.fast else []
 cmds=[
  ['-m','ece571.problem1','--data',a.data,'--city','Davis','--task','sky',*flag],
  ['-m','ece571.problem1','--data',a.data,'--city','Davis','--task','regime',*flag],
  ['-m','ece571.problem2','--data',a.data,'--city','Davis','--mode','same_city',*flag],
  ['-m','ece571.problem2','--data',a.data,'--mode','cross_city',*flag],
  ['-m','ece571.problem3','--data',a.data,'--city','Davis',*flag],
  ['-m','ece571.problem4','--data',a.data,'--city','Davis','--task','classification',*flag],
  ['-m','ece571.problem5','--data',a.data,'--source','Davis','--target','Amherst',*flag],
 ]
 for c in cmds: print('\n>>>',sys.executable,*c); subprocess.run([sys.executable,*c],check=True)
 print('\nAll tabular experiments completed. For the sequence model run: python -m ece571.problem2 --data data.xlsx --city Davis --mode sequence --fast')
if __name__=='__main__': main()
