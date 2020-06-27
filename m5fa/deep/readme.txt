# Docker build

- docker build -t mytf docker

# On ppc
# Depending on what is your bas image one of the following
- docker build -t mytfcpu docker_ppc
- docker build -t mytfgpu docker_ppc

# running a script
- ./rxxx script.py

- xxx should hold the configuration for your machine. (e.g ben, work, mlac, ...)

# Check the progress of your machine
- http://localhost:4040
