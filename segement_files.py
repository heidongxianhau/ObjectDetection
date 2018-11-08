
# coding: utf-8

# In[14]:

import numpy as np  
import os  
import tensorflow as tf  
import datetime
import time
from matplotlib import pyplot as plt  
from PIL import Image
import glob


# In[15]:

MODEL_NAME="./"
PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'  


# In[20]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# In[21]:

test_img_base_path="./Sample"
imgs_files=os.path.join(test_img_base_path,"*","*.png")
imgs_list=glob.glob(imgs_files)
num_imgs=len(imgs_list)
print("Images num:"+str(num_imgs))
inference_path="./inference_result"
new_files=[]
if not os.path.exists(inference_path):
    os.mkdir(inference_path)
total_time = 0


# In[22]:

with detection_graph.as_default():  
  with tf.Session(graph=detection_graph) as sess: 
    image_tensor = detection_graph.get_tensor_by_name('ImageTensor:0')   
    prediction = detection_graph.get_tensor_by_name('SemanticPredictions:0')  
    start_time=datetime.datetime.now()
    print("STARTING ...")
    for image_path in imgs_list:
        image_np = Image.open(image_path)
        image_np_expanded = np.expand_dims(image_np, axis=0)  
        # Definite input and output Tensors for detection_graph  
        out_name=os.path.join(inference_path,image_path.split("/")[-2],image_path.split("/")[-1])
        time1 = time.time()
        prediction_out= sess.run(  
          prediction,feed_dict={image_tensor: image_np_expanded}) 
        time2 = time.time()
        total_time += float(time2-time1)
        result=Image.fromarray(np.array(prediction_out[0]*200).astype(np.uint8))
        if  not os.path.exists(os.path.join(inference_path,out_name.split("/")[-2])):
            os.mkdir(os.path.join(inference_path,out_name.split("/")[-2]))
        result.save(out_name)
    end_time=datetime.datetime.now()
    
    print("START TIME :"+str(start_time))
    print("END TIME :"+str(end_time))
    print("THE TOTAL TIME COST IS:"+str(total_time))
    print("THE average TIME COST IS:"+str(float(total_time)/float(num_imgs)))
















