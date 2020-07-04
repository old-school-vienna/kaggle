# Docker build

- docker build -t mytf docker
## on mlac
- nohup docker build -t mytfcpu dockerppc/ > nohupb.out 2>&1 &

# On ppc
# Depending on what is your bas image one of the following
- docker build -t mytfcpu docker_ppc
- docker build -t mytfgpu docker_ppc

# running a script
- ./rxxx script.py

- xxx should hold the configuration for your machine. (e.g ben, work, mlac, ...)

# Check the progress of your machine
- http://localhost:4040


# Analyse m5 'Sales5_Ab2011_InklPred.csv'
complete
count = 60.034.810 
Days = 1 - 1969 
sales = 0 - 763 (0)

train (sales not null)
count = 58.327.370
Days = 1 - 1913

train (sales null)
prediction
count = 1.707.440
Days = 1914 - 1969 (56 = 2 * 28)
Days validation = 1914 - 1941 
Days evaluation = 1942 - 1969 

