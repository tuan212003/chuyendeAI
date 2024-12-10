import cv2
import os
from mtcnn import MTCNN

def create_img_to_video(path_video, path_folder_save,sl):
    # Mở video
    video = cv2.VideoCapture(path_video)

    tong = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Kiểm tra xem video có mở thành công không
    if not video.isOpened():
        print("Không thể mở video.")
        return

    # Tạo thư mục lưu ảnh nếu chưa tồn tại
    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)

    ds = []  # Danh sách lưu các frame

    for i in range(0,tong,(tong*2)//sl):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            print("Kết thúc video hoặc không thể đọc frame.")
            break
        h,w,_=frame.shape
        imgcrop,conf=processing(frame,w,h)
        if imgcrop is not None:
            ds.append([float(conf),imgcrop])
    
        
    ds= sorted(ds, key=lambda x: x[0], reverse=True)

    for i in range(0,min(sl,len(ds)),1):
        img=ds[i][1]
        save_path_img = os.path.join(path_folder_save, f"high_conf_{i+1}.jpg")
        cv2.imwrite(save_path_img, img)

    video.release()  # Giải phóng video
    print(f"Lưu {sl} anh vào thư mục {path_folder_save} thành công.")



def create_img_to_img(path_folder_img, path_folder_save, sl):
    # Tạo thư mục lưu ảnh nếu chưa tồn tại
    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)

    # Lấy danh sách các tệp ảnh trong thư mục
    img_files = [f for f in os.listdir(path_folder_img) if f.endswith(('jpg', 'jpeg', 'png'))]
    ds = []  # Danh sách lưu các ảnh đã xử lý

    for img_file in img_files:
        img_path = os.path.join(path_folder_img, img_file)
        img = cv2.imread(img_path)
        h,w,_=img.shape

        if img is None:
            print(f"Không thể đọc ảnh {img_path}")
            continue

        imgcrop, conf = processing(img,w,h)
        if imgcrop is not None:
            ds.append([float(conf), imgcrop])

    # Sắp xếp danh sách theo độ tin cậy (confidence) giảm dần
    ds = sorted(ds, key=lambda x: x[0], reverse=True)

    # Lưu ảnh theo yêu cầu
    for i in range(min(sl, len(ds))):
        img = ds[i][1]
        save_path_img = os.path.join(path_folder_save, f"high_conf_{i+1}.jpg")
        cv2.imwrite(save_path_img, img)

    print(f"Lưu {sl} ảnh vào thư mục {path_folder_save} thành công.")


def processing(img,w,h):
    model = MTCNN()
    results = model.detect_faces(img)
    if results:
        for result in results:
            x, y, width, height = result['box']
            conf = result['confidence']
            x1=x-w//10
            x2=x+width+w//10
            y1=y-h//10
            y2=y+height+h//10
            if x1<0:
                x1=x
            if x2>w:
                x2=x+width
            if y1<0:
                y1=y
            if y2>h:
                y2=y+height
            imgcrop=img[y1:y2,x1:x2]
            return imgcrop, conf
    else: return None,0

    

        