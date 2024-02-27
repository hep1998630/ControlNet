import replicate
from PIL import Image 
from cog import File, Path 
import cv2 
import base64
from PIL import Image


def prepare_image(frame, encode_quality= 100): 
    frame = cv2.resize(frame,(960,540))
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    _,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,encode_quality])
    dashboard_img = base64.b64encode(buffer).decode()

    return dashboard_img

image_cv= cv2.imread('tmp/out/img1.png')
image_base64= prepare_image(image_cv)
# from replicate import File 

# img_path= r"https://i.ibb.co/gSYHPR2/uliana-koliasa-fk-UD91-Ee-Cok-unsplash.jpg"
# pil_image = Image.open("tmp/out/img1.png")
# print(pil_image.size)


output = replicate.run(
    "hep1998630/controlnet:f6194f1edd8426914dc113050598d8f6c1810055fdfd3582c8045144492e5b2a",
    input={
        "input_image_string": image_base64,
        "prompt": "Put furniture in the room",
        "eta": 0,
        "seed": 3.5,
        "scale": 9,
        "a_prompt": "best quality, extremely detailed",
        "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        "strength": 1,
        "ddim_steps": 20,
        "guess_mode": False,
        "num_samples": 1,
        "image_resolution": 512,
        "detect_resolution": 512
    }
)
print(output)