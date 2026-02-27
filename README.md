# Multi_DDPP
#### In this work, we firstly trained a big dataset including differet cell assays but they represent similar task and diverse molecules. Then we transferred the knowledge from the big dataset into our specific-task and high-fidelity dataset, which comprehensively improve our model. And we constructed models with physiological parameters not just chemical information.
![图片](https://github.com/user-attachments/assets/4c3a1859-a053-47d8-bfcb-71033336f58c)
# Overview
##### * features : construction of molecular graph<br>
##### * data: datasets for models<br>
##### * Multi_DDPP :the pre-trained model on the big dataset and the model based on knowledge distillation<br>
##### * regression_model :regression models<br>
##### * Example :data saved for classification and regression<br>
##### * Prediction :Predict data without labels<br>
# Environment
  ##### * python 3.8<br>
  ##### * pytorch 2.3.0<br>
  ##### * rdkit 2021.09.2<br>
  ##### * scikit-learn 1.3.2<br>
  ##### * dgl 2.2.1<br>
# Installation
##### * Clone the current repo<br>
```
git clone https://github.com/YUZhang-utu/Macro_permeability
```
```
cd Macro_permeability
```
##### * Create a new environment with all required packages<br>
```
conda env create -f environment.yml -n macro_p
```
# Predict new molecules
##### * Save data as same as files in Example
##### * Classification prediction
```
python -m Prediction.classification_predict --checkpoint_path classification_model.ckpt --input_csv input_file.csv --output_csv result.csv
```
##### * Regression prediction
```
python Prediction/regression_predict.py --checkpoint_path regression_model.ckpt --input_csv new_data.csv
```
