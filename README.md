# 2019ccf
2019CCF BDCI-多人种人脸识别-top1方案与部分代码
压缩包格式：

![Image text](https://github.com/themostnewone/2019ccf/blob/master/img-folder/1.png)
![add image](https://github.com/themostnewone/master/2019ccf/image-folder/1.png)
![add](https://github.com/themostnewone/2019ccf/blob/master/image-folder/1.png)


按照 环境要求.txt 安装环境
按照 模型地址.txt 下载预训练模型到对应文件夹。

代码文件夹结构：

 
其中各文件夹：
blendsub:初始空文件夹，存放每个模型产生的概率文件
gfmodel:里面存放了3种模型，每个文件夹里面存放对应的预训练模型和相关代码
testing:存放数据集Test_data文件夹与submission_template.csv

操作步骤：
3种模型可以分别执行。
gfmodel/insightface/arcface.py
gfmodel/lvface/lvface.py
gfmodel/proba/proba.py
修改的部分包括3个路径，mo选择模型：
 
最后执行blend.py可在当前路径生成submit.csv提交。
