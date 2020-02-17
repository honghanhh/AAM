# [MI3.05: Research Course - Deep Learning]

### Implementation

1. Generate the dataset
```
python organize_cohn_kanade_dataset.py ./raw  ./dataset
```
```
python shape.py  ../dataset/
```
```
python shape_list.py ../dataset/
```
```
python aam_pca_train.py -O ../save_model/ ../dataset/
```
```
python aam_pca_test.py ../save_model.pkl
```
------------

Student Name: TRAN Thi Hong Hanh