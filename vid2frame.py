import cv2
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image

generator = StableDiffusion(
    img_height=1024,
    img_width=1024,
    jit_compile=False,
)


vidcap = cv2.VideoCapture('big_buck_bunny_720p_10mb.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
  img = generator.generate(
    "oil painting by monet",
    num_steps=75,
    unconditional_guidance_scale=1.5,
    temperature=1,
    batch_size=1,
    input_image= "frames/frame%d.jpg" % count
  )
  Image.fromarray(img[0]).save("out/output%d.png" % count)  



  success,image = vidcap.read()
  print('Read a new frame: %d ' % count, success)
  count += 1
