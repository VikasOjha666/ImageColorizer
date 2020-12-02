from tkinter import *
import tkinter.filedialog as filedialog
from PIL import ImageTk,Image
from utils import colour_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter.messagebox as messagebox


class AppGUI:
#Init GUI.
    def __init__(self):
        #Some vars.
        self.is_opened=0


        self.root=Tk()

        #Width and height of window.
        self.root.geometry("270x270")
        self.root.title('Image Colorization App')

        #Place canvas to display the picture.
        self.canvas=Canvas(self.root,width=256,height=256)
        self.canvas.pack(expand=YES, fill=BOTH)        

        #Menubar
        self.menubar=Menu(self.root)
        #File Menu.
        self.filemenu=Menu(self.menubar,tearoff=0)

        #New submenu.
        self.filemenu.add_command(label='New',command=self.new_file)
        #Open submenu.
        self.filemenu.add_command(label='Open',command=self.open_file)

        #Save submenu.
        self.filemenu.add_command(label='Save',command=self.save_file)

        #Save as submenu.
        self.filemenu.add_command(label='Save as',command=self.saveas_file)

        #Exit submenu.
        self.filemenu.add_command(label='Exit',command=self.root.quit)


        #Menu Separtor.
        self.filemenu.add_separator()

        #Adding the filemenu to menu bar.
        self.menubar.add_cascade(label='File',menu=self.filemenu)


        #Edit Menu
        self.editmenu=Menu(self.menubar,tearoff=0)

        #Colorize.
        self.editmenu.add_command(label='Colorize',command=self.colorize_function)

        #Grayscale.
        self.editmenu.add_command(label='Grayscale',command=self.grayscale_function)

        #Adding the editmenu to menu bar.
        self.menubar.add_cascade(label='Edit',menu=self.editmenu)


        #Help.
        self.helpmenu=Menu(self.menubar,tearoff=0)

        #HelpMenu.
        self.helpmenu.add_command(label='About',command=self.about_function)

        #Adding the editmenu to menu bar.
        self.menubar.add_cascade(label='Help',menu=self.helpmenu)


        self.root.config(menu=self.menubar)
        self.filename=''

    def new_file(self):
        self.is_opened=0
        self.canvas.delete("all")
        self.root.update()
    def open_file(self):
        #File open dialog.
        filename=filedialog.askopenfilename()
        self.filemenu=filename
        try:
            #Loading image using PIL.
            self.img=Image.open(filename)
            #Resizing to fir our canvas.
            self.img=self.img.resize((270,270))
            #Displaying image over our canvas.
            self.photoimg = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0,0,image=self.photoimg, anchor="nw")
            self.is_opened=1
        except Exception as e:
            print(e)
            print('Cancelled.')

    def save_file(self):
        if self.is_opened==1:
            img=np.array(self.img)
            cv2.imwrite(self.filename,img)
        else:
            messagebox.showerror('Error','Please open a file first')

    def saveas_file(self):
        if self.is_opened==1:
            filename=filedialog.asksaveasfilename()
            self.filename=filename
            img=Image.open('temp_img.jpg')
            img.save(self.filename)
        else:
            messagebox.showerror('Error','Please open a file first')

    def colorize_function(self):
        if self.is_opened==1:
            img=self.img
            img=np.array(img)
            img=colour_image(img)
            img=np.squeeze(img,axis=0)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig('temp_img.jpg',bbox_inches='tight')
            self.img=Image.open('temp_img.jpg')
            self.img=self.img.resize((270,270))
            self.photoimg = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0,0,image=self.photoimg, anchor="nw")
            self.root.update()
    
        else:
            messagebox.showerror('Error','Please open a file first')
    def grayscale_function(self):
        if self.is_opened==1:
            self.img=cv2.cvtColor(np.array(self.img),cv2.COLOR_RGB2GRAY)
            self.img=Image.fromarray(self.img)
            self.photoimg = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0,0,image=self.photoimg, anchor="nw")
            self.root.update()
        else:
             messagebox.showerror('Error','Please open a file first')


    def about_function(self):
       messagebox.showinfo('ABOUT','This app was devloped by Vikas Kumar Ojha \n just as a demo for image colorization using GAN')






