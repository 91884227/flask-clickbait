# flask-clickbait

###### tags: `README` 

# Notes
* The `requirements.txt` file should list all Python libraries that your notebooks depend on, and they will be installed using:
    ```
    pip install -r requirements.txt
    ```
* The folder [data](https://drive.google.com/drive/folders/15BDjL2IaX3eYdFVzT422VwCb743Hrbi3) should be downloaded to the working directory.


# flask_clickbait.py
execute code
```
python flask_clickbait.py
```
then you can open  `http://127.0.0.1:5000/` to detect clickbait

usage: 

`http://127.0.0.1:5000/title?title=居家檢疫攏是假？名醫「一張照片打臉」網罵：難怪找不出感染源`


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

