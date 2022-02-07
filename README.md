# CapsVI-Macias 
A Novel Deep Capsule Neural Network for Vowel Imagery Patterns from EEG signals

This project classifies vowel imagery CSP-EEG signals from DaSalla dataset [1] using a Capsule Neural Network denominated CapsVI. It contains the example code to train and test the model. Also, we include the signals we used in .csv format to download, those correspond to the autors of [1].

Juan A. Ramirez-Quintana, Jose M. Macias-Macias, Graciela Ramirez-Alonso,
Mario I. Chacon-Murguia, Luis F. Corral-Martinez, A Novel Deep Capsule Neural Network for Vowel Imagery Patterns from EEG signals,
Biomedical Signal Processing and Control.

To run CapsVI follow the next instructions:
  1. Download the repository of CapsVI in a zip file.
  2. Extract the files.
  3. The .csv files contain the CSP and labels for each subject, the files are located in Files folder.
  4. Open CapsVI-Macias.py in python and run it.
  5. Inside CapsVI-Macias.py are the instructions to change subject signals and labels.
  6. You can comment or uncomment lines from the code to see predictions of the network.

Also, there are some requirements to run the program:
  1. The project was made in python 3.7
  2. The packages that we used are the next ones:

  1. cudatoolkit               10.0.130                      0
  2. cudnn                     7.6.5                cuda10.0_0
  3. future                    0.18.2                   py37_0
  4. keras                     2.2.4                         0    anaconda
  5. keras-applications        1.0.8                      py_0
  6. keras-base                2.2.4                    py37_0    anaconda
  7. keras-preprocessing       1.1.2              pyhd8ed1ab_0    conda-forge
  8. matplotlib                3.1.2                    py37_1    conda-forge
  9. matplotlib-base           3.1.2            py37h2981e6d_1    conda-forge
  10. numpy                     1.18.1           py37h93ca92e_0
  11. numpy-base                1.18.1           py37hc3f5095_1
  12. numpydoc                  0.9.2                      py_0
  13. seaborn                   0.10.0                     py_0    anaconda
  14. scikit-image              0.16.2           py37h47e9c7a_0    anaconda
  15. scikit-learn              0.23.2           py37h47e9c7a_0
  16. scipy                     1.3.2            py37h29ff71c_0
  17. tensorboard               1.13.1           py37h33f27b4_0
  18. tensorflow                1.13.1          gpu_py37h83e5d6a_0
  19. tensorflow-base           1.13.1          gpu_py37h871c8ca_0
  20. tensorflow-estimator      1.13.0                     py_0
  21. tensorflow-gpu            1.13.1               h0d30ee6_0
 
It is important to say that the project was developed through spyder in anaconda, using TensorFlow 1.13.1.

For any questions, you can contact us by email: jose08m3@gmail.com and juan.rq@chihuahua.tecnm.mx

Also, you can check another version of this project [2].

This work was supported by the Tecnologico Nacional de México under
grants TecNM.

[1] C. S. DaSalla, H. Kambara, M. Sato, Y. Koike, Single-trial classification
of vowel speech imagery using common spatial patterns, Neural Networks
22 (9) (2009) 1334–1339, brain-Machine Interface. doi:https:
//doi.org/10.1016/j.neunet.2009.05.008.

[2] J. Macias-Macias, J. Ramirez-Quintana, G. Ramirez-Alonso,
M. Chacon-Murguia, Deep learning networks for vowel speech imagery
(2020) 1–6doi:10.1109/CCE50788.2020.9299143.
