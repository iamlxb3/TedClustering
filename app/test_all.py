import sys
import subprocess
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

n_cluster = [x for x in range(4, 6)]
#for n in n_cluster:
subprocess.call(['python', 'clustering.py', '-f', 'tf', '-c', 'MiniBatchKMeans', '--n_cluster', '3'],
             shell=True, stdout=subprocess.PIPE)