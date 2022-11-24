#
#
#   App
#
#

import torch
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
from model import Model
from common import get_test_transform
from utils import convert_I_to_L


@st.cache(allow_output_mutation=True)
def get_ap_vs_pa_model():
    return Model.load_from_checkpoint("runs/ap_vs_pa/9/epoch=0-step=313.ckpt").eval()


@st.cache(allow_output_mutation=True)
def get_frontal_vs_lateral_model():
    return Model.load_from_checkpoint("runs/frontal_vs_lateral/0/epoch=3-step=1252.ckpt").eval()


st.set_page_config(layout="wide")

st.markdown(f""" 
    <style> 
        .appview-container .block-container {{
            max-width: 1600px;
        }}
        .appview-container .main .block-container {{
            padding-top: 2rem;    
        }}
    </style>    
""", unsafe_allow_html=True)

st.title("TFM - Rubén Salas")

goal = st.selectbox("Objetivo", ["Frontal vs Lateral", "AP vs PA"])

img_bytes = st.file_uploader("Subir radiografía", type=["png", "jpg", "jpeg"])

if goal == "AP vs PA":
    model = get_ap_vs_pa_model()
elif goal == "Frontal vs Lateral":
    model = get_frontal_vs_lateral_model()
else:
    raise NotImplementedError()

if img_bytes is not None:
    img = Image.open(img_bytes)

    if img.mode == "I":
        img = convert_I_to_L(img)
    else:
        img = img.convert("L")

    transform = get_test_transform(model.transfer_learning_model)
    tensor_img = transform(img)

    with torch.no_grad():
        logits = model(tensor_img.unsqueeze(0))

    probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()[0]

    if goal == "AP vs PA":
        fig = go.Figure([
            go.Bar(x=["AP", "PA"], y=probs * 100, marker_color=["#636EFA", "#EF553B"])
        ])
    elif goal == "Frontal vs Lateral":
        fig = go.Figure([
            go.Bar(x=["Frontal", "Lateral"], y=probs * 100, marker_color=["#00CC96", "#AB63FA"])
        ])
    else:
        raise NotImplementedError()

    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="Probabilidad (%)", title_x=0.5)

    col_1, col_2 = st.columns(2)

    with col_1:
        st.image(img, use_column_width=True)

    with col_2:
        st.plotly_chart(fig, use_container_width=True)