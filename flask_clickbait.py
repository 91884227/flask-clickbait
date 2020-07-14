#!/usr/bin/env python
# coding: utf-8

# In[4]:


from model import ABS
# ABS("男子陳屍桃園大圳 身體腫脹難辨身分")


# In[ ]:


from flask import Flask
from flask import request
app = Flask(__name__)

### Define Discriminator system
@app.route('/', methods=['GET'])
def home():
    return "<h1>welcome to title discriminator</h1>"

    
@app.route('/title', methods=['GET'])
def get_title():
    if('title' in request.args):
        print(str(request.args['title']))
        score = ABS(str(request.args['title']))
#         if( score > 0.575):
#             buf = "clickbait %.4f" % score
#         else:
#             buf = "non-clickbait %.4f" % score

        buf = score 
        out_string = "<h1>"+str(buf)+"</h1>"
        return out_string
    else:
        return "<h1>wrong input</h1>"

if __name__ == '__main__':
    app.run(port=5000)


# In[ ]:





# In[ ]:




