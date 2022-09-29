---
sort: 1
---
# Installation 
## Preparing Environment

### With [Anaconda](https://www.anaconda.com/)
We currently provide conda environment that we test our development code on. So we can guarantee that our code will work with this environment. However, environment might contain unneccessary packages.


```console
(base) user@server:~$ git clone https://github.com/dlrudco/SADACO.git
(base) user@server:~$ cd SADACO
(base) user@server:~$ conda create -n sadaco -f sadaco_env.yml
(base) user@server:~$ conda activate sadaco
(sadaco) user@server:~$
```

### With pip
[How to install pip - Ubuntu](https://linuxize.com/post/how-to-install-pip-on-ubuntu-20.04/)

We also provide requirements.txt that you can try installing with the pip. Unfortunately, we do not guarantee full functionality and might have some of the missing packages. 

```console
user@server:~$ git clone https://github.com/dlrudco/SADACO.git
user@server:~$ cd SADACO
user@server:~$ python -m pip install -r requirements.txt
```

## Installing SADACO

Once you are done setting the environment, install SADACO with pip or setup.py

RUN
<pre><code>python setup.py develop</code></pre>
or
<pre><code>python -m pip install -e .</code></pre>

We are now on a development phase. So we recommend installing in a development mode so changes on the source codes can be immediately reflected to the package.

After the installation, you can import sadaco in your own project like below.

```console
(sadaco) user@server:~$ python
Python 3.8.10 (default, Jun  4 2021, 15:09:15) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sadaco
>>> sadaco.__version__
'SADACO version v0.1'
```

## Usage examples
[Python Pipelines](Training%26Inference.md)

[Web](../SADACO_WEB/Tutorial.md)

[Local GUI]()