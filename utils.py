from time import sleep 
from PIL import ImageGrab 
import tkinter as tk
import threading

class RecArea: 
    def __init__(self): 
        self.__start_x, self.__start_y, self.__end_x, self.__end_y = 0, 0, 0, 0 

        self.__win = tk.Tk() 
        self.__win.attributes("-alpha", 0.5) # 设置窗口半透明 
        self.__win.attributes("-fullscreen", True) # 设置全屏 
        self.__win.attributes("-topmost", True) # 设置窗口在最上层 

        self.__width, self.__height = self.__win.winfo_screenwidth(), self.__win.winfo_screenheight() 

        # 创建画布 
        self.__canvas = tk.Canvas(self.__win, width=self.__width, height=self.__height, bg="gray") 

        self.__win.bind('<Button-1>', self.button_press) # 绑定鼠标左键点击事件 
        self.__win.bind('<ButtonRelease-1>', self.button_release) # 绑定鼠标左键点击释放事件 
        self.__win.bind('<B1-Motion>', self.button_move) # 绑定鼠标左键点击移动事件 
        self.__win.bind('<Escape>', lambda e: self.__win.destroy()) # 绑定Esc按键退出事件 

        self.__win.mainloop() # 窗口持久化 

    def button_press(self, event): 
        self.__start_x, self.__start_y = event.x, event.y 

    def button_release(self, event): 
        self.__end_x, self.__end_y = event.x, event.y 
        self.__win.destroy()
        self.__win.quit()

    def button_move(self, event): 
        if event.x == self.__start_x or event.y == self.__start_y: 
            return 
        self.__canvas.delete("prscrn") 
        self.__canvas.create_rectangle(self.__start_x, self.__start_y, event.x, event.y, 
        fill='white', outline='red', tag="prscrn") 

        self.__canvas.pack() 

    def get_rec_loc(self): 
        return self.__start_x, self.__start_y, self.__end_x, self.__end_y 


class ScreenCapThread(threading.Thread): 
    def __init__(self, start_x, start_y, end_x, end_y, img_file='imgs/tmp.png',thread_name='thread'): 
        threading.Thread.__init__(self) 
        self.start_x=start_x 
        self.start_y=start_y 
        self.end_x=end_x 
        self.end_y=end_y 
        self.img_file=img_file
        self.thread_name=thread_name
        self.flag=True

    def run(self):
        self.flag=True
        while self.flag:
            im = ImageGrab.grab((self.start_x, self.start_y, self.end_x, self.end_y))
            print(self.thread_name)
            sleep(0.5)
    
    def stop(self):
        self.flag=False

def set_rec():
    prScrn = RecArea() 
    start_x, start_y, end_x, end_y = prScrn.get_rec_loc()
    print(start_x, start_y, end_x, end_y)
    return (start_x, start_y, end_x, end_y)

def create_thread(rec,name='thread'):
    thread=ScreenCapThread(rec[0], rec[1], rec[2], rec[3],thread_name=name)
    return thread

def create_rec_thread(name='thread'):
    start_x, start_y, end_x, end_y=set_rec()
    thread=ScreenCapThread(start_x, start_y, end_x, end_y,thread_name=name)
    return thread

if __name__ == '__main__': 
    thread1=create_rec_thread('thread1')
    thread2=create_rec_thread('thread2')
    thread3=create_rec_thread('thread3')

    thread1.start()
    thread2.start()
    thread3.start()
