## GANI: <u>G</u>lobal <u>A</u>ttacks via <u>N</u>ode <u>I</u>njections

The relevant codes for "**GANI: Global Attacks on Graph Neural Networks via Imperceptible Node Injections**". [[arXiv](https://arxiv.org/abs/2210.12598)]

<p align="center"> <img src="fig/framework.png" /> <p align="center"><em>Fig. 1. A systematic framework of GANI and corresponding evaluations. The red marks indicate the corresponding generated fake node including both features and neighbors. The colors of nodes represent the classes, and the cloud-shaped circle means a wrong classification of the node.</em></p>



- ### Requirements

  - torch == 1.8.0

  - deeprobust == 0.2.1

  - Other packages will be installed together when installing deeprobust

    

- ### Code illustrations 

  - **data**: Folder to save the generated adversarial data after attacks.

  - **dataset**: Folder of clean datasets.

  - **ori_model**: Folder of original models trained from the clean data.

  - **ga_homophily.py**: The genetic algorithm for neighbor selection of GANI.

  - **main.py**: Examples for using GANI to achieve node injection attacks.

  - **node_injection.py**: Main codes for GANI.

  - **utils.py**: Main codes for evaluation.

    

- ### Run the demo

  ```
  python main.py
  ```

  

- ### Cite

  If you find this work is helpful, please cite our paper, Thank you.

  ```
  TBA
  ```

  