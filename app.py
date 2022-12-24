#
#
#   App
#
#

import os
import torch
import fastdl
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
from config import DATA_ROOT_DIR
from utils import convert_I_to_L
from torchvision import transforms
from models import ClassificationModel
from common import get_test_transform
from torch.nn.functional import mse_loss
from api import APPAClassifier, FrontalLateralClassifier, ChestXrayOrNotClassifier


@st.cache(allow_output_mutation=True)
def get_ap_vs_pa_model():
    return APPAClassifier()


@st.cache(allow_output_mutation=True)
def get_frontal_vs_lateral_model():
    return FrontalLateralClassifier()


@st.cache(allow_output_mutation=True)
def get_chest_or_not_model():
    return ChestXrayOrNotClassifier()


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

goal = st.selectbox("Objetivo", ["Frontal vs Lateral", "AP vs PA", "Tórax o no"])

img_bytes = st.file_uploader("Subir radiografía", type=["png", "jpg", "jpeg"])

if goal == "AP vs PA":
    model = get_ap_vs_pa_model()
elif goal == "Frontal vs Lateral":
    model = get_frontal_vs_lateral_model()
elif goal == "Tórax o no":
    model = get_chest_or_not_model()
else:
    raise NotImplementedError()


if img_bytes is not None:
    img = Image.open(img_bytes)

    if img.mode == "I":
        img = convert_I_to_L(img)
    else:
        img = img.convert("L")

    if goal in ("AP vs PA", "Frontal vs Lateral"):
        results = model.predict(img)

        if goal == "AP vs PA":
            fig = go.Figure([
                go.Bar(x=["AP", "PA"], y=results["proba"] * 100, marker_color=["#636EFA", "#EF553B"])
            ])
        elif goal == "Frontal vs Lateral":
            fig = go.Figure([
                go.Bar(x=["Frontal", "Lateral"], y=results["proba"] * 100, marker_color=["#00CC96", "#AB63FA"])
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
    elif goal == "Tórax o no":
        results = model.predict(img)

        _, col_2, col_3, _ = st.columns(4)

        with col_2:
            diff_error = results["error_threshold"] - results["error"]
            st.metric("Error de reconstrucción", f"{results['error']:.5f}",
                      delta=-diff_error, delta_color="inverse")

        with col_3:
            st.metric("¿Es radiografía de tórax?", "Sí" if diff_error > 0 else "No")

        col_1, col_2 = st.columns(2)

        with col_1:
            st.write("Original")
            st.image(img.resize((256, 256), Image.Resampling.LANCZOS), use_column_width=True)

        with col_2:
            st.write("Reconstrucción")
            st.image(results["img"], use_column_width=True)
