# mobile price prediction app
# i made this for my ml assignment

import gradio as gr
import pickle
import pandas as pd

# loading my model that i trained before
file = open("mobile_priced_model.pkl", 'rb')
model = pickle.load(file)
file.close()

# this function will predict the price range
def predict_price(battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi):
    
    # create dataframe with all features
    data = {
        'battery_power': [battery_power], 'blue': [blue], 'clock_speed': [clock_speed],
        'dual_sim': [dual_sim], 'fc': [fc], 'four_g': [four_g], 'int_memory': [int_memory],
        'm_dep': [m_dep], 'mobile_wt': [mobile_wt], 'n_cores': [n_cores], 'pc': [pc],
        'px_height': [px_height], 'px_width': [px_width], 'ram': [ram], 'sc_h': [sc_h],
        'sc_w': [sc_w], 'talk_time': [talk_time], 'three_g': [three_g],
        'touch_screen': [touch_screen], 'wifi': [wifi],
        'total_camera': [pc + fc], 'screen_size': [sc_h * sc_w],
        'pixels': [px_height * px_width], 'battery_per_weight': [battery_power / (mobile_wt + 1)]
    }
    features = pd.DataFrame(data)
    
    # get prediction
    prediction = model.predict(features)[0]
    
    # return price range
    price_ranges = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
    return price_ranges[prediction]


inputs = [
    gr.Slider(500, 2000, value=1000, step=50, label="Battery Power (mAh)"),
    gr.Radio([0, 1], value=1, label="Bluetooth (0=No, 1=Yes)"),
    gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="Clock Speed (GHz)"),
    gr.Radio([0, 1], value=1, label="Dual SIM (0=No, 1=Yes)"),
    gr.Slider(0, 20, value=5, step=1, label="Front Camera (MP)"),
    gr.Radio([0, 1], value=1, label="4G (0=No, 1=Yes)"),
    gr.Slider(2, 128, value=32, step=2, label="Internal Memory (GB)"),
    gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Mobile Depth (cm)"),
    gr.Slider(80, 200, value=150, step=5, label="Weight (g)"),
    gr.Slider(1, 8, value=4, step=1, label="Number of Cores"),
    gr.Slider(0, 20, value=10, step=1, label="Primary Camera (MP)"),
    gr.Slider(500, 2000, value=1000, step=50, label="Pixel Height"),
    gr.Slider(500, 2000, value=1000, step=50, label="Pixel Width"),
    gr.Slider(256, 4096, value=2048, step=256, label="RAM (MB)"),
    gr.Slider(5, 20, value=10, step=0.5, label="Screen Height (cm)"),
    gr.Slider(0, 20, value=5, step=0.5, label="Screen Width (cm)"),
    gr.Slider(2, 20, value=10, step=1, label="Talk Time (hours)"),
    gr.Radio([0, 1], value=1, label="3G (0=No, 1=Yes)"),
    gr.Radio([0, 1], value=1, label="Touch Screen (0=No, 1=Yes)"),
    gr.Radio([0, 1], value=1, label="WiFi (0=No, 1=Yes)"),
]
# put all inputs in list
# output
output = gr.JSON(label="Prediction Results")


# create the interface
app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs='text',
    title="Mobile Price Range Classifier",
    description="Enter mobile phone specifications to predict the price range",
)

# run the app
app.launch(share=True)