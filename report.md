## Index
- [I - Background removal](#i-background-removal)
- [II - Data augmentation](#ii-data-augmentation)
- [III - Machine learning](#iii-machine-learning)
- [IV - Deep learning](#iv-deep-learning)
- [V - Web part and deployment](#v-web-part-and-deployment)

# I - Background removal

### First approach

```py
lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
lab[:,:,0] = lab[:,:,0]/10 # L
lab[:,:,1] += np.where(lab[:,:,1] > 125, 140, lab[:,:,1]) # A
lab[:,:,2] = lab[:,:,2]/10 # B

# We apply filter on 'A' and downscale 'L' and 'B'
new_rgb_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# We apply a np.where with alpha mask to remove background
a_mask = pcv.rgb2gray_lab(rgb_img=new_rgb_img, channel='a')
a_mask = np.where(a_mask <= int(a_mask.mean()), 0, a_mask)
a_mask = np.where(a_mask > 0, 1, a_mask)

# At this point we have a great mask but there are some noise around the plant

# We find all the contours in the mask and we fill the holes
cnts = cv2.findContours(a_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(a_mask, cnts, (255,255,255))
result = cv2.bitwise_and(rgb_img,rgb_img,mask=a_mask)

masked_image = pcv.apply_mask(img=gray_img, mask=a_mask, mask_color='black')
```

Results:
....

At this point we have a great mask and the background removal work fine but only with healthy plants. If we have a plant with a disease we have a problem because the mask is not good enough.

### Second approach

Our second approach is to use k-means clustering to find the background and the plant. We use the same image as before.
And a lot of logical operations to remove the background.
The k-means work fine but it took a lot of time to process the image.

Results:
...


### Last program

At the end we have refocus on the first approach with something more simple. With colors mask on hsv channels.

```py
color_dict_HSV = {
    'black': [[180, 255, 26], [0, 0, 0]],
    'white': [[180, 38, 255], [0, 0, 166]],
    'gray': [[180, 38, 166], [0, 0, 25]],
    'red1': [[180, 255, 255], [155, 50, 40]],
    'red2': [[9, 255, 255], [0, 50, 70]],
    'pink1': [[6, 178, 255], [0, 0, 26]],
    'pink2': [[180, 178, 255], [175, 0, 26]],
    'pink3': [[176, 255, 255], [155, 38, 25]],
    'orange': [[25, 255, 255], [5, 38, 191]],
    'brown': [[25, 255, 191], [5, 50, 25]],
    'yellow': [[40, 255, 255], [15, 15, 10]],
    'yellowgreen': [[60, 255, 250], [30, 100, 200]],
    'green': [[85, 255, 255], [41, 15, 10]],
    'bluegreen': [[90, 255, 255], [76, 38, 25]],
    'blue': [[127, 255, 255], [91, 38, 25]],
    'purple': [[155, 255, 255], [128, 38, 25]],
    'lightpurple': [[155, 128, 255], [128, 38, 25]],
}
```

Results:
....


### Conclusion

We have a lot of different approach to remove the background. The first one is the best but it's not perfect. We have to find a way to improve it.
An improvement proposal would have been to build a dataset using the correct masked images. And to train a U-net neural network to segment the leaves. We would have had better results on the whole dataset.

# II - Data augmentation

....

# III - Machine learning

### Models

### Data preprocessing

### Results

# IV - Deep learning


### Models


We have try different models to classify the images.

- RESNET50
- RESNET50V2
- INCEPTIONRESNETV2
- INCEPTIONV3
- EFFICIENTNETB0
- CLASSIC CNN
- CONVNEXT
- LAB AND HSV PROCESS [see this link](https://github.com/joaopauloschuler/two-path-noise-lab-plant-disease)
- VGG16 / VGG19



### Data preprocessing


We have try different data preprocessing techniques.

- Image augmentation
- Image normalization
- Image resizing
- Image cropping
- Image without background



### Results


<table>
  <tr>
    <th>Ranking</th>
    <th>Model</th>
    <th>Preprocessing</th>
    <th>Number of images per classes</th>
    <th>Accuracy (validation)</th>
    <th>Size image</th>
    <th>Time (in minutes)</th>
  </tr>

  <tr>
    <td>1 🥇</td>
    <td>CONVNEXT</td>
    <td></td>
    <td>1100</td>
    <td>0.9882</td>
    <td>224x224</td>
    <td>150</td>
  </tr>
  <tr>
    <td>2 🥈</td>
    <td>INCEPTIONV3</td>
    <td></td>
    <td>3000</td>
    <td>0.9801</td>
    <td>128x128</td>
    <td>60</td>
  </tr>
  <tr>
    <td>3 🥉</td>
    <td>RESNET50</td>
    <td></td>
    <td>1100</td>
    <td>0.9717</td>
    <td>128x128</td>
    <td>20</td>
  </tr>
  <tr>
    <td>4</td>
    <td>EFFICIENTNETB7</td>
    <td></td>
    <td>1100</td>
    <td>0.9729</td>
    <td>128x128</td>
    <td>108</td>
  </tr>
  <tr>
    <td>0</td>
    <td>VGG16</td>
    <td></td>
    <td>1100</td>
    <td>0.9707</td>
    <td>128x128</td>
    <td>20</td>
  </tr>
  <tr>
    <td>0</td>
    <td>XCEPTION</td>
    <td></td>
    <td>1100</td>
    <td>0.9686</td>
    <td>128x128</td>
    <td>53</td>
  </tr>
  <tr>
    <td>0</td>
    <td>EFFICIENTNETB0</td>
    <td></td>
    <td>1100</td>
    <td>0.9641</td>
    <td>128x128</td>
    <td>25</td>
  </tr>

  <tr>
    <td>0</td>
    <td>LAB_PROCESS</td>
    <td>Image normalization + lab images</td>
    <td>1000</td>
    <td>0.9651</td>
    <td>128x128</td>
    <td>50</td>
  </tr>

  <tr>
    <td>0</td>
    <td>INCEPTIONV3</td>
    <td>min_max image normalization</td>
    <td>1300</td>
    <td>0.9583</td>
    <td>128x128</td>
    <td>110</td>
  </tr>

  <tr>
    <td>0</td>
    <td>CONVNEXT</td>
    <td>Image normalization</td>
    <td>1100</td>
    <td>0.9594</td>
    <td>128x128</td>
    <td>65</td>
  </tr>

  <tr>
    <td>0</td>
    <td>INCEPTIONRESNETV2</td>
    <td></td>
    <td>1100</td>
    <td>0.9548</td>
    <td>128x128</td>
    <td>86</td>
  </tr>

   <tr>
    <td>0</td>
    <td>RESNET50V2</td>
    <td></td>
    <td>1100</td>
    <td>0.9419</td>
    <td>128x128</td>
    <td>33</td>
  </tr>


  <tr>
    <td>0</td>
    <td>INCEPTIONV3</td>
    <td></td>
    <td>1100</td>
    <td>0.939</td>
    <td>128x128</td>
    <td>34</td>
  </tr>


  <tr>
    <td>0</td>
    <td>CLASSIC_CNN</td>
    <td></td>
    <td>1100</td>
    <td>0.8087</td>
    <td>128x128</td>
    <td>45</td>
  </tr>

  <tr>
    <td>0</td>
    <td>ALEXNET</td>
    <td></td>
    <td>1100</td>
    <td>0.7979</td>
    <td>128x128</td>
    <td>20</td>
  </tr>
</table>

### CONVNEXT

### RESNET50

### INCEPTIONV3

### Conclusion


# V - Web part and deployment
