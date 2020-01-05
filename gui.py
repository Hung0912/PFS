import PIL
from tkinter import filedialog, Text, messagebox
import os
from PIL import Image, ImageTk
import tkinter as tk
import picture_fuzzy_rules as PFR
from picture_fuzzy_clustering import PFC, afterClusterData
from readImage import readImage, data2Image

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

def runPFR():
        global label_input_image, label_output_image, frame_output
        label_output_image.destroy()
        if haveInput == False:
                popup('Please select your input image first!')  
        option = var.get()
        if option == 0:
                popup('Please select option to run!')
        elif option == 1:
                # print(image_input)
                PFR.main(image_input)
                img = Image.open('output.jpg')
                img = ImageTk.PhotoImage(img)
                label_output_image = tk.Label(frame_output, image = img)
                label_output_image.image = img
                label_output_image.pack()
        else:
                PFR.main1(image_input)
                if option == 2:    
                        img = Image.open('max_result.jpg')
                        img = ImageTk.PhotoImage(img)
                        label_output_image = tk.Label(frame_output, image = img)
                        label_output_image.image = img
                        label_output_image.pack()
                if option == 3:
                        img = Image.open('mean_result.jpg')
                        img = ImageTk.PhotoImage(img)
                        label_output_image = tk.Label(frame_output, image = img)
                        label_output_image.image = img
                        label_output_image.pack()

def runPFC():
        global label_input_image, label_output_image, frame_output
        label_output_image.destroy()
        if haveInput == False:
                popup('Please select your input image first!')  
        else:
                # print(image_input)
                data = readImage(image_input)
                result_membership_matrix, result_cluster_centers = PFC(data)
                after_data = afterClusterData(data, result_membership_matrix, result_cluster_centers)
                image = data2Image(after_data.reshape((384,512,3)))
                img = ImageTk.PhotoImage(image)
                label_output_image = tk.Label(frame_output, image = img)
                label_output_image.image = img
                label_output_image.pack()

def sel():
        selection = "You selected the option " + str(var.get())
        print(selection)

def clear():
        global label_input_image, label_output_image, var
        label_input_image.destroy()
        label_output_image.destroy()
        var.set(0)

var = tk.IntVar()

frame = tk.Frame(root, padx = 10, pady = 10)
frame_input = tk.Frame(frame, highlightbackground="red", highlightcolor="red", highlightthickness=1, width = 512, height = 384, padx = 5, pady = 5)
frame_menu = tk.LabelFrame(frame, text = "Menu")
frame_output = tk.Frame(frame, highlightbackground="red", highlightcolor="red", highlightthickness=1, width = 512, height = 384, padx = 5, pady = 5)


button_select_image = tk.Button(frame_menu, text = 'Select image', width = 20, pady = 5, command = lambda:open_image()).pack(padx = 30, pady = 10)
button_clear = tk.Button(frame_menu, text = 'Clear', width = 20, pady = 5, command = lambda:clear()).pack(padx = 30, pady = 10)

frame_PFR = tk.LabelFrame(frame_menu, text="Picture Fuzzy Rule")
frame_PFR.pack(fill="both", expand="yes")

R1 = tk.Radiobutton(frame_PFR, text="Defuzzy output", variable=var, value=1,
                  command=sel).pack( anchor = tk.W )

R2 = tk.Radiobutton(frame_PFR, text="Max output", variable=var, value=2,
                  command=sel).pack( anchor = tk.W )

R3 = tk.Radiobutton(frame_PFR, text="Mean output", variable=var, value=3,
                  command=sel).pack( anchor = tk.W)

frame_PFC = tk.LabelFrame(frame_menu, text="Picture Fuzzy Clustering")
frame_PFC.pack(fill="both", expand="yes")
PFC_var = tk.StringVar()
PFC_label = tk.Label( frame_PFC, textvariable=PFC_var, relief=tk.RAISED, bd = 0)
PFC_var.set("Clusters K = 8, max_iter = 1000,\nm = 2.00, alpha = 0.5")
PFC_label.pack(expand = "yes")
button_runPFC = tk.Button(frame_PFC, text = 'Run PFC', width = 20, pady = 5, command = lambda:runPFC()).pack(padx = 30, pady = 10)


button_runPFR = tk.Button(frame_PFR, text = 'Run PFR', width = 20, pady = 5, command = lambda:runPFR()).pack(padx = 30, pady = 10)
button_quit= tk.Button(frame_menu, text = 'Quit', width = 20, pady = 5, command = root.quit).pack(padx = 30, pady = 10)

label_input_image = tk.Label()
label_output_image = tk.Label()

frame.pack()
frame_input.pack(side = tk.LEFT)
frame_menu.pack(side = tk.LEFT, fill="both", expand="yes")
frame_output.pack(side = tk.RIGHT)

root.mainloop()
    
