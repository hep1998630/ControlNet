import cv2 
import base64

def prepare_image(frame, encode_quality= 50): 
    frame = cv2.resize(frame,(960,540))
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    _,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,encode_quality])
    dashboard_img = base64.b64encode(buffer).decode()

    return dashboard_img