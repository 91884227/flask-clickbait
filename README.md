# flask-clickbait

###### tags: `README` 

# flask_clickbait.py
execute code
```
python flask_clickbait.py
```
then you can open  `http://127.0.0.1:5000/` to detect clickbait
usage: 

`http://127.0.0.1:5000/title?title=這是充滿炭14的實驗室`


# data_preprocessing.py
usage:
```python
from data_preprocessing import encode
encode("韓國打贏日本")
# return: [354, 4234, 354]
```

# model.py
usage:
```python
from model import ABS
ABS("男子陳屍桃園大圳 身體腫脹難辨身分")
# return: 0.03051474690437317
```

