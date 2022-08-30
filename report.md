# Background removal

# Deep learning


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
    <td>0</td>
    <td>CONVNEXT</td>
    <td></td>
    <td>1100</td>
    <td>0.9849</td>
    <td>128x128</td>
    <td>50</td>
  </tr>
  <tr>
    <td>0</td>
    <td>CONVNEXT</td>
    <td></td>
    <td>4500</td>
    <td>0.9879</td>
    <td>64x64</td>
    <td>120</td>
  </tr>
  <tr>
    <td>0</td>
    <td>CONVNEXT</td>
    <td></td>
    <td>750</td>
    <td>0.9815</td>
    <td>224x224</td>
    <td>80</td>
  </tr>
  <tr>
    <td>2 🥈</td>
    <td>RESNET50</td>
    <td></td>
    <td>1100</td>
    <td>0.9717</td>
    <td>128x128</td>
    <td>20</td>
  </tr>
  <tr>
    <td>3 🥉</td>
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