# TFM
Trabajo de Fin de Máster

## Installation
1. Install dependencies
```console
$ pip install -r requirements.txt
```
2. Download datasets and pretrained-weights
```console
$ bash download.sh
```

## Usage
### Models
#### Chest Xray or Not
**Train**
```console
$ python train_chest_xray_or_not.py
```
**Test**
```console
$ python test_chest_xray_or_not.py
```

#### Frontal vs Lateral
**Train**
```console
$ python train_frontal_vs_lateral.py
```
**Test**
```console
$ python test_frontal_vs_lateral.py
```

#### AP vs PA
**Train**
```console
$ python train_ap_vs_pa.py
```

**Test**
```console
$ python test_ap_vs_pa.py
```

### Interactive visualization
```console
$ streamlit run app.py
```

### Python API
#### Chest Xray or Not
```python
from api import ChestXrayOrNotClassifier

clf = ChestXrayOrNotClassifier()
results = clf.predict(<path_of_your_img>) 
```

#### Frontal vs Lateral
```python
from api import FrontalLateralClassifier

clf = FrontalLateralClassifier()
results = clf.predict(<path_of_your_img>) 
```

#### AP vs PA
```python
from api import APPAClassifier

clf = APPAClassifier()
results = clf.predict(<path_of_your_img>) 
```
