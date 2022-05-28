import tkinter as tk
import tkinter.font as font
import ttkbootstrap as ttkb
from PIL import Image,ImageTk
import utils

main_page_bg_path='bg.png'

style=ttkb.Style(theme='litera')
root=style.master
sw=root.winfo_screenwidth()//2
sh=root.winfo_screenheight()//2

main_page_window_size=(400,600)

def set_window_position(root,pos):
    ws=pos
    root.geometry(str(ws[0])+'x'+str(ws[1]))
    root.geometry('+{}+{}'.format(sw-ws[0]//2,sh-ws[1]//2))
    root.resizable(0,0)

def resize(image,k):
    size=image.size
    return image.resize((int(size[0]*k),int(size[1]*k)))

class Main_Page:
    def __init__(self,root):
        self.root=root
        self.root.title('扑克牌工具')
        
        image=Image.open(main_page_bg_path)
        image=resize(image,0.2)
        bg=ImageTk.PhotoImage(image)
        cv =ttkb.Canvas(self.root,width=main_page_window_size[0],height=main_page_window_size[1])
        cv.pack()
        cv.create_image(main_page_window_size[0]//2,main_page_window_size[1]//2,image=bg)
        
        ttkb.Label(
            self.root,
            text='欢迎使用扑克牌工具',
            bootstyle="dark",
            font=font.Font(size=20)
            ).place(anchor='n',relx=0.5,rely=0.05)

        opt = [
            ("是", 1),
            ("否", 2)]
        self.v = tk.IntVar()
        self.v.set(1)
        ttkb.Label(
            self.root,
            text='是否是地主',
            bootstyle='dark'
        ).place(anchor='n',relx=0.35,rely=0.15)

        tk.Radiobutton(
            self.root,
            text='是',
            variable=self.v,
            value=1
            ).place(anchor='n',relx=0.5,rely=0.15)

        tk.Radiobutton(
            self.root,
            text='否',
            variable=self.v,
            value=2
            ).place(anchor='n',relx=0.6,rely=0.15)
        
        self.area1=None
        self.area2=None
        self.area3=None

        self.thread1=None
        self.thread2=None
        self.thread3=None

        ttkb.Button(
            self.root,
            text='设置你的卡牌区域',
            bootstyle='dark',
            command=lambda: self.set_area(1)
        ).place(anchor='n',relx=0.5,rely=0.25)

        ttkb.Button(
            self.root,
            text='设置玩家1的卡牌区域',
            bootstyle='dark',
            command=lambda: self.set_area(2)
        ).place(anchor='n',relx=0.5,rely=0.35)

        ttkb.Button(
            self.root,
            text='设置玩家2的卡牌区域',
            bootstyle='dark',
            command=lambda: self.set_area(3)
        ).place(anchor='n',relx=0.5,rely=0.45)

        ttkb.Button(
            self.root,
            text='开始',
            bootstyle='dark',
            command=self.start
        ).place(anchor='n',relx=0.5,rely=0.55)

        ttkb.Button(
            self.root,
            text='暂停',
            bootstyle='dark',
            command=self.pause
        ).place(anchor='n',relx=0.5,rely=0.65)

        self.root.protocol("WM_DELETE_WINDOW", self.exit)
        self.root.mainloop()
    
    def get_opt(self):
        print(self.v.get())

    def set_area(self,area):
        if area==1:
            self.area1=utils.set_rec()
        if area==2:
            self.area2=utils.set_rec()
        if area==3:
            self.area3=utils.set_rec()
    
    def start(self):
        self.thread1=utils.create_thread(self.area1,name='area1')
        self.thread2=utils.create_thread(self.area2,name='area2')
        self.thread3=utils.create_thread(self.area3,name='area3')

        self.thread1.start()
        self.thread2.start()
        self.thread3.start()

    def pause(self):
        self.thread1.stop()
        self.thread2.stop()
        self.thread3.stop()

        self.thread1.join()
        self.thread2.join()
        self.thread3.join()

    def exit(self):
        self.pause()

        self.root.destroy()
        self.root.quit()

set_window_position(root,main_page_window_size)
Main_Page(root)