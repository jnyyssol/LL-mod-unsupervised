# LL-mod-unsupervised

# LL mod: Unsupervised paper
This repository is based on a snapshot of the [LogLead](https://github.com/EvoTestOps/LogLead) tool. Please see the full tool for more details including the download links for datasets.

This version is only intended to work as a replication package for the paper [Speed and Performance of Parserless and Unsupervised Anomaly Detection Methods
on Software Logs](https://arxiv.org/abs/2312.01934) (link is preprint). This version has reduced functionality but also some specific additions we didn't want to include in the general tool.  

To run, install the required dependencies. The supplied env files can be installed with conda. Then download the data and set the path in `unsupervised_models.py`. After that you can run it. Please be aware of the `frac_data` variable that sets the amount of data used. Large amount of data will likely run out of memory in home computers.  