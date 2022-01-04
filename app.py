import gradio as gr
import numpy as np
import torch
from modality_lstm import ModalityLSTM
import torch.nn as nn
from helper import score_to_modality
from PIL import Image

label_mapping = {
        'car': [0,'images/Cars.jpg'],
        'walk': [1,'images/walk.jpg'],
        'bus': [2,'images/bus.jpg'],
        'train': [3,'images/train.jpg'],
        'subway': [4,'images/subway.jpg'],
        'bike': [5,'images/bike.jpg'],
        'run': [6,'images/walk.jpg'],
        'boat': [7,'images/walk.jpg'],
        'airplane': [8,'images/walk.jpg'],
        'motorcycle': [9,'images/walk.jpg'],
        'taxi': [10,'images/taxi.jpg']
    }

def pred(dist,speed,accel,timedelta,jerk,bearing,bearing_rate):



    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_on_gpu = False
    output_size = 5
    hidden_dim = 128
    trip_dim = 7
    n_layers = 2
    drop_prob = 0.2
    net = ModalityLSTM(trip_dim, output_size, batch_size, hidden_dim, n_layers, train_on_gpu, drop_prob, lstm_drop_prob=0.2)
    net.load_state_dict(torch.load("Model_Wieghts"))
    net.eval()

    a=torch.tensor([[dist,speed,accel,timedelta,jerk,bearing,bearing_rate]])
    a=a.float()
    a=a.unsqueeze(0)
    l = torch.tensor([1]).long()
    b,c=net(a,l)
    b=b.squeeze(0)
    b=score_to_modality(b)
    b=b[0]
    print(b)
    for k,v in label_mapping.items():
        if b == v[0]:
            return (str(k),Image.open(v[1]))



















def greet(name):
  return "Hello " + name + "!!"

iface = gr.Interface(fn=pred, inputs=['number',"number","number",'number',"number","number","number"], outputs=["text",gr.outputs.Image(type="pil")])
iface.launch()