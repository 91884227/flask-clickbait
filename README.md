# flask-clickbait

###### tags: `README` 

# flask_clickbait.py
執行以下程式
```
python flask_clickbait.py
```
即可在本地端 `http://127.0.0.1:5000/` 執行
使用方式: e.g. 

`http://127.0.0.1:5000/title?title=這是充滿炭14的實驗室`


# data_preprocessing.py
使用方式:
```python
from data_preprocessing import encode
encode("韓國打贏日本")
# return: [354, 4234, 354]
```

# model.py
使用方式:
```python
from model import ABS
ABS("男子陳屍桃園大圳 身體腫脹難辨身分")
# return: 0.03051474690437317
```

