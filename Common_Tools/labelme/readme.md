
##Lableme  


    labelme文件，主要的功能是通过标注或者网络预测的mask，转换成labelme可以自动识别的格式。  

主要包含**五个步骤**:  
+ 将原始的Image按照labelme的编码方式，生成imageData 每个原始image.jpg图像对应一个.image.json文件。.'/img2json.py'  
+ 获取预测或者标注mask文件的轮廓:读取mask.png图像，将RGB(BGR)转为灰度图，通过设置阈值，将灰度图转为二值图.  
+ 腐蚀较小的区域，特别是在预测的mask中非常常见。单独的多个点，导入到labelme中，会导致labelme闪崩。  
+ 通过opencv的findContours获取图像轮廓（可能多个）.'./get_contours.py'  
+ 由于模型预测的mask轮廓点非常密集，因此需要将相邻点进行删除，以达到简化修改调整的目标。  


###使用方法:  

1. ./img2json.py 输入需要处理图像的文件夹和保存生成的.json文件夹 process_img_path="/home/xxh/Desktop/tissue/images/", out_file_path="/home/xxh/Desktop/tissue/json/"
2. ./imitate_json_optimize.py   参数分别是：原始图像生成的imageData的.json文件对应的文件夹，融合后的json文件保存的文件夹，原始图像文件夹，mask文件夹.img_json_path="/home/xxh/Desktop/tissue/json/", fusion_path="/home/xxh/Desktop/tissue/fusion/",imgPath="/home/xxh/Desktop/tissue/images/",maskPath='/home/xxh/Desktop/tissue/mask/',