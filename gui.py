import PIL
from tkinter import filedialog, Text, messagebox
import os
from PIL import Image, ImageTk
import tkinter as tk
import picture_fuzzy_rules as PFR

haveInput = bool(False)
image_input = str()

root = tk.Tk()
root.title('GUI')
# root.geometry("300x280+300+300")

def popup(message):
        tk.messagebox.showwarning('This is warning',message)

def select_image():
        filename = filedialog.askopenfilename(initialdir="/", title = "Select File", filetypes=(("executables","*.jpg"),("all files", "*.*")))
        # print('1')
        return filename

def open_image():
        global frame_input, label_input_image, haveInput, image_input
        label_input_image.destroy()
        image_input = select_image()
        img = Image.open(image_input)
        img = img.resize((512, 384), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        label_input_image = tk.Label(frame_input, image = img)
        label_input_image.image = img
        label_input_image.pack()
        haveInput = True

def run():
        global label_input_image, label_output_image, frame_output
        label_output_image.destroy()
        if haveInput == False:
                popup('Please select your input image first!')
        else:
                print(image_input)
                PFR.main(image_input)
                img = Image.open('output.jpg')
                img = ImageTk.PhotoImage(img)
                label_output_image = tk.Label(frame_output, image = img)
                label_output_image.image = img
                label_output_image.pack()

frame = tk.Frame(root, padx = 10, pady = 10)
frame_input = tk.Frame(frame, highlightbackground="red", highlightcolor="red", highlightthickness=1, width = 512, height = 384, padx = 1, pady = 1)
frame_menu = tk.LabelFrame(frame, bd = 0)
frame_output = tk.Frame(frame, highlightbackground="red", highlightcolor="red", highlightthickness=1, width = 512, height = 384, padx = 5, pady = 5)

button_select_image = tk.Button(frame_menu, text = 'Select image', width = 20, pady = 5, command = lambda:open_image()).pack(padx = 30, pady = 10)
button_run = tk.Button(frame_menu, text = 'Run PFR', width = 20, pady = 5, command = lambda:run()).pack(padx = 30, pady = 10)
button_quit= tk.Button(frame_menu, text = 'Quit', width = 20, pady = 5, command = root.quit).pack(padx = 30, pady = 10)

label_input_image = tk.Label()
label_output_image = tk.Label()

frame.pack()
frame_input.pack(side = tk.LEFT)
frame_menu.pack(side = tk.LEFT)
frame_output.pack(side = tk.RIGHT)

root.mainloop()
    
