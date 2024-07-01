* 1 `python -u pre_cluster.py <config path>` for two level pre-cluster 
* 2 `cd models; python -u clam.py <config path>` for warmup MIL model
* 3 `python -u HIS.py <config path>` for Hierarchical Instance Searching
* 4 `python -u project.py <config path>` for Instance classifier(projector)
* 5 `python -u refinement.py <config path>` for WSI Instance refinement