<h1>
    <img src="https://user-images.githubusercontent.com/38484045/155925781-fe7795d9-4e7c-439d-bbb3-ac4888f8254c.png" width=80 height=80/> 
    SADACO : Swift Audio DAta COmprehension Framework + Stethoscope Audio DAtaset Collections
</h1> 

Welcome to SADACO repository! Where spooky chest sounds are brought to life.

!!! Web Server for deploying sadaco service is available at [SADACO-WEB](https://github.com/dlrudco/SADACO_WEB) !!!

We currently provide basic train-test-explain pipelines for [ICBHI](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge), [Fraiwan](https://data.mendeley.com/datasets/jwyy9np4gv/3), [PASCAL-Heartsound](http://www.peterjbentley.com/heartchallenge/) and bunch of other NN APIs.


**** We are still under development process. Let us know if there's any question or suggestion. We'll try to make it happen :) ****

**** Detailed READMEs and Documentations are also coming ****

## Installation

RUN
<pre><code>python setup.py develop</code></pre>
or
<pre><code>python -m pip install -e .</code></pre>
------

## Getting Started

*Currently, we provide the training and validating code for only the ICBHI dataset.*
First move on to the pipelines folder and,
RUN
<pre><code>python ICBHI.py --conf-file configs/train_basic.yml</code></pre>

This will start running ResNet50 training on ICBHI official split dataset with Banlanced Batch Sampler.

You can also try different settings by running 
<pre><code>python ICBHI.py --conf-file configs/train_contrastive.yml</code></pre>

or changing values in the yaml files, creating new configuration that suits your intention.

------


## Contributors ✨

Thanks goes to these wonderful people([rolls](https://allcontributors.org/docs/en/emoji-key) shown by icons):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/dlrudco"><img src="https://avatars0.githubusercontent.com/u/37071556?v=4" width="100px;" alt=""/><br /><sub><b>dlrudco</b></sub></a><br /><a href="https://github.com/dlrudco/steth-audio/commits?author=dlrudco" title="Rolls">🤔💻📖🚧📦💬🔬</a></td>
    <td align="center"><a href="https://github.com/lunayht"><img src="https://avatars1.githubusercontent.com/lunayht" width="100px;" alt=""/><br /><sub><b>lunayht</b></sub></a><br /><a href="https://github.com/dlrudco/steth-audio/commits?author=lunayht" title="Rolls">🤔💻📖📦💬🔬</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

------

## Inspiring References
1. [Supervised Contrastive Loss](https://github.com/HobbitLong/SupContrast)

