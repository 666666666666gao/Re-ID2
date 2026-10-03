import json,os,random,sys,time
rng=random.Random(20261003)
with open(sys.argv[1],'x') as output:
 for i in range(80):
  output.write(json.dumps(dict(index=i,value=rng.random(),pid=os.getpid(),pgid=os.getpgrp(),time=time.time()))+'\n')
  output.flush()
  time.sleep(.02)
