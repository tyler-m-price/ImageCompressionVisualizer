import dearpygui.dearpygui as dpg
from PIL import Image
import numpy as np

def update_filepath(sender, app_data):
    current_val = dpg.get_value("filepath")
    
dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

def file_select_callback(sender, app_data):
    print(f"Selected file: {app_data['file_path_name']}")
    global selected_file
    selected_file = app_data['file_path_name']

with dpg.file_dialog(directory_selector=False, show=False, callback=file_select_callback, id="my_file_dialog"):
    dpg.add_file_extension(".*")

with dpg.window(label="Config Menu", width = 500, height = 500, pos=(500,0)):
    dpg.add_button(label="Select File", callback=lambda: dpg.show_item("my_file_dialog"))
    dpg.add_text("Hello World")
    
filepath = "default.png" 
width, height, channels, data = dpg.load_image(filepath)
with dpg.texture_registry(): 
    dpg.add_static_texture(width, height, data, tag="ImageTexture")
    
with dpg.window(label="Image", width = 500, height = 1000, pos=(0,0)):
    dpg.add_image("ImageTexture")
    
with dpg.window(label="Compression Visualizer", width = 500, height = 500, pos=(500,500)):
    filepath = "default.png" 
    img = Image.open(filepath)
    array = np.array(img)
    dpg.add_text(array)
    dpg.add_text("Hello World")
    
dpg.create_viewport(title='Image Compression Visualizer', width=1000, height=1000)    
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
