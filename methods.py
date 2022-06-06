from PIL import ImageGrab 
import tkinter as tk
import threading
import os
import sys

import time
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_img_size,  colorstr, cv2,
                           increment_path, non_max_suppression,  scale_coords, strip_optimizer, xyxy2xywh)
from utils.datasets import LoadImages
from models.common import DetectMultiBackend
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.no_grad()
class Run:
    def __init__(self):
        self.flag = True
        self.flag_tip=True
        self.rec=None

    def tip(self):
        self.flag_tip=False

    def stop(self):
        self.flag=False
        self.flag_tip=False
    
    def grab(self):
        im = ImageGrab.grab((self.rec[0][0], self.rec[0][1], self.rec[0][2], self.rec[0][3]))
        im.save('imgs/area1.png')
        im = ImageGrab.grab((self.rec[1][0], self.rec[1][1], self.rec[1][2], self.rec[1][3]))
        im.save('imgs/area2.png')
        im = ImageGrab.grab((self.rec[2][0], self.rec[2][1], self.rec[2][2], self.rec[2][3]))
        im.save('imgs/area3.png')

    def run(
        self,
        weights=ROOT / 'playcard_3.pt',  # model.pt path(s)
        source=ROOT / 'imgs',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'mydata.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.1,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        self.flag=True
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        while self.flag:
            while self.flag_tip:
                pass
            # Dataloader
            self.grab()
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # Inference
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + \
                        ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        f=open(txt_path + '.txt', 'w')
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        f.close()
                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        cv2.imwrite(save_path, im0)

                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
            self.flag_tip=True

workspace=os.getcwd()

class RecArea: 
    def __init__(self,master): 
        self.__start_x, self.__start_y, self.__end_x, self.__end_y = 0, 0, 0, 0 
        self.__win = tk.Toplevel(master)
        self.__width, self.__height = self.__win.winfo_screenwidth(), self.__win.winfo_screenheight() 
        self.__win.attributes("-alpha", 0.5) # 设置窗口半透明 
        self.__win.attributes("-fullscreen", True) # 设置全屏 
        self.__win.attributes("-topmost", True) # 设置窗口在最上层 

        # 创建画布 
        self.__canvas = tk.Canvas(self.__win, width=self.__width, height=self.__height, bg="gray") 

        self.__win.bind('<Button-1>', self.button_press) # 绑定鼠标左键点击事件 
        self.__win.bind('<ButtonRelease-1>', self.button_release) # 绑定鼠标左键点击释放事件 
        self.__win.bind('<B1-Motion>', self.button_move) # 绑定鼠标左键点击移动事件 
        self.__win.bind('<Escape>', lambda e: self.exit()) # 绑定Esc按键退出事件 

        self.__win.mainloop() # 窗口持久化 

    def button_press(self, event): 
        self.__start_x, self.__start_y = event.x, event.y 

    def button_release(self, event): 
        self.__end_x, self.__end_y = event.x, event.y 
        self.exit()

    def button_move(self, event): 
        if event.x == self.__start_x or event.y == self.__start_y: 
            return 
        self.__canvas.delete("prscrn") 
        self.__canvas.create_rectangle(self.__start_x, self.__start_y, event.x, event.y, 
        fill='white', outline='red', tag="prscrn") 

        self.__canvas.pack() 
    
    def exit(self):
        self.__win.quit()
        self.__win.destroy()

    def get_rec_loc(self): 
        return self.__start_x, self.__start_y, self.__end_x, self.__end_y 

class ModelThread(threading.Thread): 
    def __init__(self): 
        threading.Thread.__init__(self) 
        self.r=Run()

    def run(self):
        try:
            if sys.platform == "win32":
                os.system('echo y|rd /s runs/detect/exp')
            else:
                os.system('rm -r runs/detect/exp')
        except:
            pass
        self.r.run()

    def tip(self):
        self.r.tip()

    def stop(self):
        self.r.stop()

def set_rec(master):
    prScrn = RecArea(master) 
    start_x, start_y, end_x, end_y = prScrn.get_rec_loc()
    print(start_x, start_y, end_x, end_y)
    return (start_x, start_y, end_x, end_y)

def create_model_thread():
    thread=ModelThread()
    return thread