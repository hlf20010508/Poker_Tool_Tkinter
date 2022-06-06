import tkinter as tk
import tkinter.font as font
import ttkbootstrap as ttkb
from PIL import Image,ImageTk
import methods

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

        self.cap_thread=None
        self.model_thread=None

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

        self.label1=ttkb.Label(
            self.root,
            text='你的卡牌：',
            bootstyle='dark'
        )
        self.label1.place(anchor='n',relx=0.5,rely=0.3)

        self.label2=ttkb.Label(
            self.root,
            text='玩家1的卡牌：',
            bootstyle='dark'
        )
        self.label2.place(anchor='n',relx=0.5,rely=0.4)

        self.label3=ttkb.Label(
            self.root,
            text='玩家2的卡牌：',
            bootstyle='dark'
        )
        self.label3.place(anchor='n',relx=0.5,rely=0.5)

        ttkb.Button(
            self.root,
            text='开始',
            bootstyle='dark',
            command=self.start
        ).place(anchor='n',relx=0.5,rely=0.55)

        ttkb.Button(
            self.root,
            text='提示',
            bootstyle='dark',
            command=self.tip
        ).place(anchor='n',relx=0.5,rely=0.65)

        ttkb.Button(
            self.root,
            text='停止',
            bootstyle='dark',
            command=self.stop
        ).place(anchor='n',relx=0.5,rely=0.75)

        self.root.protocol("WM_DELETE_WINDOW", self.exit)
        self.root.mainloop()
    
    def get_opt(self):
        print(self.v.get())

    def set_area(self,area):
        if area==1:
            self.area1=methods.set_rec(self.root)
        if area==2:
            self.area2=methods.set_rec(self.root)
        if area==3:
            self.area3=methods.set_rec(self.root)
    
    def tip(self):
        self.cap_thread=methods.create_screen_cap_thread([self.area1,self.area2,self.area3])
        self.cap_thread.start()

        t1=open('pytorch_yolo_v5/runs/detect/exp/labels/area1.txt')
        t2=open('pytorch_yolo_v5/runs/detect/exp/labels/area2.txt')
        t3=open('pytorch_yolo_v5/runs/detect/exp/labels/area3.txt')

        txt1=t1.read().strip().split('\n')
        txt2=t2.read().strip().split('\n')
        txt3=t3.read().strip().split('\n')

        t1.close()
        t2.close()
        t3.close()

        dic=['3','4','5','6','7','8','9','10','J','Q','K','A','2','BJ','CJ']
        txt1=' '.join(sorted([dic[int(i.split(' ')[0])] for i in txt1],reverse=True,key=lambda n: dic.index(n)))
        txt2=' '.join(sorted([dic[int(i.split(' ')[0])] for i in txt2],reverse=True,key=lambda n: dic.index(n)))
        txt3=' '.join(sorted([dic[int(i.split(' ')[0])] for i in txt3],reverse=True,key=lambda n: dic.index(n)))

        print(txt1)
        print(txt2)
        print(txt3)

        self.label1['text']=txt1
        self.label2['text']=txt2
        self.label3['text']=txt3

    def start(self):
        self.cap_thread=methods.create_screen_cap_thread([self.area1,self.area2,self.area3])
        self.cap_thread.start()
        self.model_thread=methods.create_model_thread()
        self.model_thread.start()

    def stop(self):
        self.model_thread.stop()
        self.model_thread.join()

    def exit(self):
        self.stop()
        self.root.destroy()
        self.root.quit()

set_window_position(root,main_page_window_size)
Main_Page(root)