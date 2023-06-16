--> The input image is located in the input folder. 
--> The output contents are located in the output folder.

Please follow the instructions below to install the conda environment and execute python scripts.

1-) Change the directory to where the project_Acar folder is located.

2-) Write the following commands on Ubuntu/Linux Terminal to create an Anaconda environment with the YAML file:

conda env create -f UACMPVSNP.yaml
conda activate UACMPVSNP

3-) You should be able to run the Python script with the following command to check whether they are working or not:

python3 motion_prediction.py 1 1

4-) Check argument table for python script below to use different configurations:

Arg1   Arg2   Model Name    Input Type    Initial ROI
0      0      yolov4-tiny   Camera        Manual
0      1      yolov4-tiny   Camera        Auto
1      0      yolov4-tiny   Video         Manual
1      1      yolov4-tiny   Video         Auto
2      0      yolov4        Camera        Manual
2      1      yolov4        Camera        Auto
3      0      yolov4        Video         Manual
3      1      yolov4        Video         Auto
4      0      haar          Camera        Manual
4      1      haar          Camera        Auto
5      0      haar          Video         Manual
5      1      haar          Video         Auto

If you give "0" as second argument then you should select initial ROI from the panel that opens.

Note #1: You can exit the code by pressing the "q" key on the keyboard for both of the scripts. Also, UACMPVSNP stands for "Utku Acar Computer Vision Project".

Note #2: If you want to use newly recorded video instead of the pre-recorded one you can run a different python script just before the actual project script(motion_prediction.py) as below:

python3 video_record.py

I have used the command pipeline below from terminal to get results with pre-recorded video easier.

python3 motion_prediction.py 1 0 && python3 motion_prediction.py 1 1 && python3 motion_prediction.py 3 0 && python3 motion_prediction.py 3 1 && python3 motion_prediction.py 5 0 && python3 motion_prediction.py 5 1