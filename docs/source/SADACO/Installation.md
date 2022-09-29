---
sort: 1
---
# Installation 
## Preparing Environment
We currently provide conda environment that we test our development code on. So we can guarantee that our code will work with this environment. However, environment might contain unneccessary packages.


```console
(base) user@server:~$ git clone https://github.com/dlrudco/SADACO.git
(base) user@server:~$ cd SADACO
(base) user@server:~$ conda create -n sadaco -f sadaco_env.yml
(base) user@server:~$ conda activate sadaco
(sadaco) user@server:~$
```