# ML-Parallelization

Collecton of tests of parallelization of data collections and ML Trainig in various platforms and applications. It includes:
  - XGBoost on Ray (XGBoost with Ray.ipynb)
  
  - XGBoost on RAY and MODIN (XGBoost with RAY and MODIN - OOM DATA.ipynb)
    - with Pandas
    - with Modin
    - with SharedRayDMatrix with (parquet) - AWS, OOM Dataset
    - OOM Test
      - with RAY, MODIN, AWS
      - Partially Successfull!
  
  - XGBoost on AWS (xgboost_abalone_dist_script_mode-ST.ipynb)
    - with AWS's Framework
    - with AWS's Built-In Algorithm
    - with AWS's Built-In Algorithm and PIPE MODE to address OOM Error - TBC

  - TensorFlow on AWS (AWS_DataPipping_TFMIrroredStrategy.ipynb)
    - DISTRIBUTED TENSORFLOW on AWS
    - DISTRIBUTED INPUT AND TRAINING
    - DATA INPUT MODES: "Pipe", "File", "FastFile"
    - TRAIN PARALLELIZATION APPROACH: "TF - Mirrored Strategy", "SMD - Sage Maker Distributed", "MPI"
    - DATA PARALLELIZATION APPROACH: data sharding (4 and 120 Shards)
    - DATA SOURCE: S3, FSX for Lustre

Recomendations, Findings, Analysis and Notes in the individual files
