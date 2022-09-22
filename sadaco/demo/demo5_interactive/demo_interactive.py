from random import sample
import tkinter as tki
from tkinter import ttk
from tkinter.messagebox import showinfo

from functools import partial
import numpy as np
from PIL import ImageTk, Image
from apis.explain.visualize import get_input_img
from apis.traintest.demo import demo_helper
from utils.config_parser import parse_config_obj
import os

LARGEFONT =("Verdana", 35)
master_config = '../demo_materials/demo_configs.yml'
model_config = '../demo_materials/demo_model.yml'
master_cfg = parse_config_obj(master_config)
model_cfg = parse_config_obj(model_config)

helper = demo_helper(master_cfg, model_cfg)

def updateDisplay(displayVar, myString):
    displayVar.set(myString)

def main():
    app = tkinterApp()
    tki.mainloop()
    

class tkinterApp(tki.Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tki.Tk.__init__(self, *args, **kwargs)
        self.geometry("960x600+100+100")
        # creating a container
        container = tki.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (InferPage, TrainPage):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(InferPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
  
# first window frame startpage
  
class InferPage(tki.Frame):
    def __init__(self, parent, controller):
        tki.Frame.__init__(self, parent)
         
        self.can_explain = False
        self.blank = np.zeros((128,128,3)).astype(np.uint8)
        img = Image.fromarray(self.blank).resize((480,480))
        stgImg1 = ImageTk.PhotoImage(img)
        self.label1= tki.Label(self, image=stgImg1)
        self.label1.grid(row=4, column=0, columnspan=4)
        self.label1.configure(image=stgImg1)
        self.label1.image = stgImg1
        stgImg2 = ImageTk.PhotoImage(img)
        self.label2= tki.Label(self, image=stgImg2)
        self.label2.grid(row=4, column=4, columnspan=4)
        self.label2.configure(image=stgImg2)
        self.label2.image = stgImg2
        self.listbox = tki.Listbox(self, height=0, selectmode='single')
        import os
        wavlist = os.listdir('../demo_materials/')
        wavlist = [w for w in wavlist if 'wav' in w]
        for w in wavlist:
            self.listbox.insert(0,w)
        self.listbox.grid(row=0, column=3, rowspan=4)
        self.method_num=tki.IntVar()
        self.class_num=tki.IntVar()
        self.sample_path = 'hey'
        
        self.displayVar = tki.StringVar()

        self.displayLab = tki.Label(self, textvariable=self.displayVar)

        self.displayLab.grid(row=1, column=0, rowspan=3, columnspan=3)
        self.btn1 = tki.Button(self, text='Inference', command=partial(self.demo_infer, self.displayVar))
        self.btn1.grid(row=0, column=0)
        
        self.btn2 = tki.Button(self, text='Explain', command=partial(self.demo_explain, self.displayVar))
        self.btn2['state'] = tki.DISABLED
        self.btn2.grid(row=0, column=4)

        self.exp_method_btn1=tki.Radiobutton(self, text="GradCAM", value=0, variable=self.method_num)
        self.exp_method_btn2=tki.Radiobutton(self, text="IntGrad", value=1, variable=self.method_num)
        self.exp_method_btn1.grid(row=2, column=4)
        self.exp_method_btn2.grid(row=2, column=5)
        
        self.cls_method_btn1=tki.Radiobutton(self, text="Normal", value=0, variable=self.class_num)
        self.cls_method_btn2=tki.Radiobutton(self, text="Wheeze", value=1, variable=self.class_num)
        self.cls_method_btn3=tki.Radiobutton(self, text="Crackle", value=2, variable=self.class_num)
        self.cls_method_btn4=tki.Radiobutton(self, text="Both", value=3, variable=self.class_num)
        self.cls_method_btn1.grid(row=3, column=4)
        self.cls_method_btn2.grid(row=3, column=5)
        self.cls_method_btn3.grid(row=3, column=6)
        self.cls_method_btn4.grid(row=3, column=7)
    
        goto_train = ttk.Button(self, text ="Train",
        command = lambda : controller.show_frame(TrainPage))
     
        # putting the button in its place by
        # using grid
        goto_train.grid(row = 5, column = 0, padx = 10, pady = 10)
        
    def demo_infer(self, displayVar):
        selection = self.listbox.curselection()
        self.sample_path = os.path.join('../demo_materials/',self.listbox.get(selection[0]))
        updateDisplay(displayVar,f"{self.sample_path}")
        self.btn2['state'] = tki.NORMAL
        inputimg = get_input_img(self.sample_path)
        result = helper.do_inference(self.sample_path)
        updateDisplay(displayVar,f"{result}")
        img = Image.fromarray(inputimg).resize((480,480))
        stgImg = ImageTk.PhotoImage(img)
        self.label1.configure(image=stgImg)
        self.label1.image = stgImg
        img = Image.fromarray(self.blank).resize((480,480))
        stgImg2 = ImageTk.PhotoImage(img)
        self.label2.configure(image=stgImg2)
        self.label2.image = stgImg2
        
    def demo_explain(self, displayVar):
        exp_img = helper.do_explanation(self.sample_path, self.method_num.get(), self.class_num.get())
        img = Image.fromarray(exp_img).resize((480,480))
        stgImg = ImageTk.PhotoImage(img)
        self.label2.configure(image=stgImg)
        self.label2.image = stgImg
  
          
  
  
# second window frame page1
class TrainPage(tki.Frame):
     
    def __init__(self, parent, controller):
         
        tki.Frame.__init__(self, parent)
        goto_infer = ttk.Button(self, text ="Infer",
        command = lambda : controller.show_frame(InferPage))
     
        # putting the button in its place by
        # using grid
        goto_infer.grid(row = 5, column = 0, padx = 10, pady = 10)
        
        self.pb = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280
        )
        self.pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)
        self.value_label = ttk.Label(self, text=self.update_progress_label())
        self.value_label.grid(column=0, row=1, columnspan=2)
        start_button = ttk.Button(
            self,
            text='Progress',
            command=self.progress
        )
        start_button.grid(column=0, row=2, padx=10, pady=10, sticky=tki.E)

        stop_button = ttk.Button(
            self,
            text='Stop',
            command=self.stop
        )
        stop_button.grid(column=1, row=2, padx=10, pady=10, sticky=tki.W)
        
    def update_progress_label(self):
        return f"Current Progress: {self.pb['value']}%"
    
    def progress(self):
        if self.pb['value'] < 100:
            self.pb['value'] += 20
            self.value_label['text'] = self.update_progress_label()
        else:
            showinfo(message='The progress completed!')
            
    def stop(self):
        self.pb.stop()
        self.value_label['text'] = self.update_progress_label()
        
    
if __name__ == "__main__":
    main()