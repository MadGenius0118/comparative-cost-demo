import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set the title and sidebar layout
st.set_page_config(page_title="Phen2Test: Comparative Cost Analysis", layout="wide")

@st.cache_data
def get_sampled_df(panel_yield_ratio, prediction_df):
    prediction_sampled_df = prediction_df.copy()
    prediction_sampled_wes_df = prediction_df[prediction_df["y_test"] == 0].sample(n=panel_yield_ratio, replace=True, random_state=7)
    prediction_sampled_panel_df = prediction_df[prediction_df["y_test"] == 1].sample(n=100 - panel_yield_ratio, replace=True, random_state=7)
    prediction_sampled_df = pd.concat([prediction_sampled_wes_df, prediction_sampled_panel_df])
    # panel_yield_ratio = st.slider("Select threthold (0 - Panel, 1 - WES):", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
    return prediction_sampled_df

    

with st.sidebar:
    st.title("Input parameter")
    # st.write("### Input Price")
    col1, col2 = st.columns(2)
    price_wes = col1.number_input("WES/WGS Price", min_value=0, max_value=500000, value=3000)
    price_panel = col2.number_input("Gene Panel Price", min_value=0, max_value=500000, value=1500)

    # st.write("### Model Performance Setting")
    model_performance_setting = st.radio("Model evaluated under",
                                        ["Columbia Medical Center",
                                        "Children's Hospital of Philadelphia Cohort"])

    if model_performance_setting == "Columbia Medical Center":
        prediction_df = pd.read_csv("Random Forest_performance_cuimc.csv")
        cols_select = ["y_test", "y_prob_0", "y_prob_1"]
        prediction_df = prediction_df[cols_select]
    else:
        prediction_df = pd.read_csv("chop_val.csv") # switch to chop
        cols_select = ["y_test", "y_prob_0", "y_prob_1"]
        prediction_df = prediction_df[cols_select]

    # st.write("### User Preference")
    # col1, col2 = st.columns(2)
    # with col1:
    #     test_selection = st.radio("Preferred Test",
    #                                     ["WES/WGS",
    #                                     "Gene Panel",
    #                                     "No preference: threshold set as 0.5"])
    # if test_selection == "No preference: threshold set as 0.5":
    #     is_disable = True
    # else:
    #     is_disable = False
    # with col2:
    #     threshold_input = st.number_input("Threshold", min_value=0.0, max_value=1.0,
    #                                     step=0.01,value = 0.5,
    #                                     disabled=is_disable)

   

    # if test_selection == "WES/WGS":
    #     prediction_df["prediction"] = np.where(prediction_df["y_prob_1"]>=threshold_input, 1, 0)
    # elif test_selection == "Gene Panel":
    #     prediction_df["prediction"] = np.where(prediction_df["y_prob_0"] >= threshold_input, 0, 1)
    # else:
    #     prediction_df["prediction"] = np.where(prediction_df["y_prob_1"] >= 0.5, 1, 0)

    # Create a slider with custom labels and values
    panel_yield_ratio = st.radio("Panel Yield Ratio:", [20, 40, 60, 80])
    n_wes = 100 - panel_yield_ratio
    n_panel = panel_yield_ratio
    prediction_sampled_df = get_sampled_df(panel_yield_ratio, prediction_df)
    threshold_input = st.slider("Select threthold (0 - Panel, 1 - WES):", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
    prediction_sampled_df["prediction"] = np.where(prediction_sampled_df["y_prob_0"] >= threshold_input, 0, 1)
    performance_dict = classification_report(prediction_sampled_df["y_test"],
                                            prediction_sampled_df["prediction"],
                                            target_names=["panel", "WES/WGS"],
                                        output_dict=True)
    recall_wes = performance_dict["WES/WGS"]["recall"]
    recall_panel = performance_dict["panel"]["recall"]
    precision_wes = performance_dict["WES/WGS"]["precision"]
    precision_panel = performance_dict["panel"]["precision"]
    performance_table = pd.DataFrame([[recall_wes, recall_panel],
                                    [precision_wes, precision_panel]], index = ["recall", "precision"],
                                    columns=["WES/WGS", "Panel"])
    st.table(performance_table)

    
    # n_wes = performance_dict["WES/WGS"]["support"]
    # n_panel = performance_dict["panel"]["support"]
    # n_wes = 60
    # n_panel = 40
    

    

# if precision_panel == 0:
#     model_pred_price = price_wes * 100
# elif precision_wes == 0:
#     model_pred_price = price_panel *100 + 60*price_wes
# else:
#     model_pred_price = price_wes * recall_wes * n_wes \
#                        + price_wes * (1-precision_wes) * n_wes +\
#                        price_panel * recall_panel*n_panel \
#                        + price_panel *(1-precision_panel) * n_panel + price_wes *(1-precision_panel)* n_panel

TP = n_wes * recall_wes
FN = n_wes - TP
if precision_wes != 0:
    FP = TP / precision_wes - TP
else:
    FP = 0
TN = n_panel - FP
st.write(f'TP: {TP}, FP:{FP}, TN: {TN}, FN: {FN}')
model_pred_price = (TP + FP) * price_wes + (TN + FN) * price_panel + FN * price_wes

WES_only_price = price_wes * 100
Panel_only_price = price_panel *100 + price_wes * (100 - panel_yield_ratio)

st.write("### Costs Per Individual")
col1, col2, col3 = st.columns(3)
col1.metric(label="Model Pred", value=f"${model_pred_price/100:,.2f}")
col2.metric(label="WES Only", value=f"${WES_only_price/100:,.2f}")
col3.metric(label="Panel Only", value=f"${Panel_only_price/100:,.2f}")

st.write("#### Savings When Using Our Predictive Model")
col1, col2 = st.columns(2)
saving_wes_per = (WES_only_price - model_pred_price) /100
saving_panel_per = (Panel_only_price - model_pred_price)/100
col1.metric(label="WES only", value=f"${saving_wes_per:,.2f}/Per patient")
col2.metric(label="Panel Only", value=f"${saving_panel_per:,.2f}/Per patient")


def create_bar_chart(a, b, c):
    fig, ax = plt.subplots()
    bars = ['Model Prediction', 'WES only', "Panel"]
    values = [a, b, c]

    ax.bar(bars, values)
    ax.set_ylabel('Total Cost')
    ax.set_title('Cost Comparison')
    return fig

# Display the bar chart
st.pyplot(create_bar_chart(model_pred_price, WES_only_price,Panel_only_price))