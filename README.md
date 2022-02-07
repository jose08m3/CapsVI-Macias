# CapsVI-Macias 
A Novel Deep Capsule Neural Network for Vowel Imagery Patterns from EEG signals

This project classify vowel imagery CSP-EEG signals from DaSalla dataset [1] using a Capsule Neural Network denomined CapsVI. It contains the example code to train and test the model. Also, we include the signals we used in .csv format to download, those correspond to the aurtors of [1].

Juan A. Ramirez-Quintana, Jose M. Macias-Macias, Graciela Ramirez-Alonso,
Mario I. Chacon-Murguia, Luis F. Corral-Martinez, A Novel Deep Capsule Neural Network for Vowel Imagery Patterns from EEG signals,
Biomedical Signal Processing and Control.

To run CapsVI follow the next instructions:
  1. Download the repository of CapsVI in a zip file.
  2. Extract the files.
  3. The .csv files contains the CSP and labels for each subject, the files are located in Files folder.
  4. Open CapsVI-Macias.py in python and run it.
  5. Inside CapsVI-Macias.py are the instructions to change subject signals and labels.
  6. You can comment or discomment lines from the code to see predictions of the network.

Also, there are some requirements to run the program:
  1. The project was made in python 3.7
  2. The packages that we used are the next ones:

cudatoolkit               10.0.130                      0
cudnn                     7.6.5                cuda10.0_0
future                    0.18.2                   py37_0
keras                     2.2.4                         0    anaconda
keras-applications        1.0.8                      py_0
keras-base                2.2.4                    py37_0    anaconda
keras-preprocessing       1.1.2              pyhd8ed1ab_0    conda-forge
matplotlib                3.1.2                    py37_1    conda-forge
matplotlib-base           3.1.2            py37h2981e6d_1    conda-forge
numpy                     1.18.1           py37h93ca92e_0
numpy-base                1.18.1           py37hc3f5095_1
numpydoc                  0.9.2                      py_0
seaborn                   0.10.0                     py_0    anaconda
scikit-image              0.16.2           py37h47e9c7a_0    anaconda
scikit-learn              0.23.2           py37h47e9c7a_0
scipy                     1.3.2            py37h29ff71c_0
tensorboard               1.13.1           py37h33f27b4_0
tensorflow                1.13.1          gpu_py37h83e5d6a_0
tensorflow-base           1.13.1          gpu_py37h871c8ca_0
tensorflow-estimator      1.13.0                     py_0
tensorflow-gpu            1.13.1               h0d30ee6_0

It is important to say that the project was developed trouhg spyder in anaconda, using tensorflow 1.13.1.

For any questions you can contact us by email: jose08m3@gmail.com and juan.rq@chihuahua.tecnm.mx

[1] C. S. DaSalla, H. Kambara, M. Sato, Y. Koike, Single-trial classification
of vowel speech imagery using common spatial patterns, Neural Networks
22 (9) (2009) 1334–1339, brain-Machine Interface. doi:https:
//doi.org/10.1016/j.neunet.2009.05.008.
