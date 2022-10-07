import cv2
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image
#from createframes import createframes

generator = StableDiffusion(
    img_height=768,
    img_width=768,
    jit_compile=False,
)



count = 454
prompt = 'oil painting by van gogh'  
prompttwo = 'oil painting by monet'  
promptselector = 0
#createframes()

while True:
  frameone = cv2.imread("frames/frame%d.jpg" % count)
  frametwo = cv2.imread("frames/frame%d.jpg" % (count + 1 ))
  difference = cv2.subtract(frameone , frametwo)
  b, g, r = cv2.split(difference)    
  if cv2.countNonZero(b) <= 700000 and cv2.countNonZero(g) <= 700000 and cv2.countNonZero(r) <= 700000 and promptselector == 0:
    print("The color is within tolerance range, so we will use the first prompt")
    print(cv2.countNonZero(b))
    print(cv2.countNonZero(g))
    print(cv2.countNonZero(r))
    img = generator.generate(
    prompt,
    num_steps=85,
    unconditional_guidance_scale=3.5,
    temperature=1,
    batch_size=1,
    input_image= "frames/frame%d.jpg" % count
    )
    Image.fromarray(img[0]).save("out/output%d.png" % count)  

  else:
    promptselector += 1
    print('The color of image is outside tolerance range, so we will switch to permanently prompt two')
    img = generator.generate(
    prompttwo,
    num_steps=85,
    unconditional_guidance_scale=3.5,
    temperature=1,
    batch_size=1,
    input_image= "frames/frame%d.jpg" % count
    )
    Image.fromarray(img[0]).save("out/output%d.png" % count)  


  


  print('Read a new frame: %d ' % (count +1))

  count += 1
