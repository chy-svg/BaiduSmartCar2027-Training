2026/10/14
20:51
使用低置信度过滤(阈值0.8)，并确定了映射表结构，便于替换
以下是第一次有无后实验处理对比
总体数据对比
指标	          无处理	后处理	   变化
识别文字总条数	    151	     108	   -43 条
有识别结果的图片数	 10	      9	        sign_2 变为空
平均置信度（估算）	0.78	 0.92	    提升
低分噪声（<0.8）	43 条	 0 条	   清除
备注：或许是置信度阈值过高，部分正确答案被当成噪声清除，建议调整阈值。



2026/10/19
19:31
进行第二次实验，调整置信度阈值，尝试减少漏检情况
实验情况:
将置信度阈值调整至0.7，情况仍不理想

进行第三次实验，使用文档方向分类 文档扭曲矫正 文本行方向分类 尝试减少误检漏检情况
实验情况:
部分图片误检漏检情况得到改善，部分图片出现图片倒置现象，下一步关闭文档方向分类

进行第四次实验，在第三次实验基础上关闭 文档方向分类
实验情况:整体得到改善
具体情况：
sign1中公交车牌中部分较小的数字(约几十像素)无法识别
sign2中误将一段远处的护栏识别成了数字64
sign3中少部分字符未被识别(倾斜角度大、位于显示屏导致不清晰、像素小)
sign4中两处较小，字符排列密集的繁体字未正确识别
sign5正确处理
sign6正确处理
sign7未能识别部分倾斜字符和密集排列字符
sign8基本正确处理，只有一处艺术加工的字符未被识别
sign9正确处理
sign10大部分字符(倾斜程度极大)未识别
结论：面临字符倾斜角度大，字符所占像素过小造成的误检漏检
命令行指令:
paddleocr ocr \
  --input /mnt/c/Users/chenmuyang/BaiduSmartCar2027-Training/data \
  --device gpu \
  --lang ch \
  --use_doc_orientation_classify False \
  --use_doc_unwarping True \
  --use_textline_orientation True \
  --save_path /root/ocr_output
使用merge_ocr_post.py脚本批处理json文件
环境:WSL2-Ubuntu24.04
Python 3.10.12







2026/10/21/
22:04
第五次实验
尝试使用 UVDoc 模型对图片进行预处理，解决文字倾斜度过大造成的漏检问题

使用UVDoc预处理图片
paddleocr text_image_unwarping \
  --model_name UVDoc \
  --input /mnt/c/Users/chenmuyang/BaiduSmartCar2027-Training/data/sign_10.jpg \
  --device gpu \
  --save_path /root/ocr_output

执行文字识别
paddleocr ocr \
  --input /mnt/c/Users/chenmuyang/Desktop/sign_10.jpg \
  --device gpu \
  --lang ch \
  --use_doc_orientation_classify False \
  --save_path /root/ocr_output

实验情况:预处理情况并不理想，拉正程度有限，文字识别有限，与第四次实验相比的确有提升，但是不大
结论:建议UVDoc适合图片扭曲造成的倾斜处理，建议使用opencv进行透视变换

2026/10/21
22:33
第六次实验
尝试使用opencv对sign10图像进行预处理
以下是预处理代码,目前仅适用于sign10

cat > /root/tilt_preprocessing.py <<'PYEOF'
import cv2
import numpy as np
import os

def tilt_preprocessing(img_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return

    # 1. 转换到 HSV 颜色空间，提取橙色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 橙色的 HSV 范围
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 2. 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤掉太小的噪点
        if area > 10000:
            count += 1
            # 找到最小外接矩形（带角度）
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算透视变换的目标点（拉平成宽200，高800的长方形）
            # 为了确保拉平方向正确，先对 4 个点排序
            rect_pts = np.zeros((4, 2), dtype="float32")
            s = box.sum(axis=1)
            rect_pts[0] = box[np.argmin(s)] # 左上
            rect_pts[2] = box[np.argmax(s)] # 右下
            diff = np.diff(box, axis=1)
            rect_pts[1] = box[np.argmin(diff)] # 右上
            rect_pts[3] = box[np.argmax(diff)] # 左下

            # 定义目标坐标（拉直后的宽高）
            w, h = 200, 800
            dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

            # 计算透视变换矩阵并应用
            M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (w, h))

            # 保存抠出来并拉平的春联
            out_file = os.path.join(output_dir, f'sign_10_cropped_{count}.jpg')
            cv2.imwrite(out_file, warped)
            print(f"已保存拉平的春联: {out_file}")

    print(f"处理完成,共处理 {count} 张倾斜图像。")

if __name__ == '__main__':
    # 定义输入输出路径
    INPUT_IMG = '/mnt/c/Users/chenmuyang/BaiduSmartCar2027-Training/data/sign_10.jpg'
    OUTPUT_DIR = '/root/ocr_output'
    
    tilt_preprocessing(INPUT_IMG, OUTPUT_DIR)
PYEOF




进行文字识别
paddleocr ocr \
  --input /mnt/c/Users/chenmuyang/Desktop/sign_10.jpg \
  --device gpu \
  --lang ch \
  --use_doc_orientation_classify False \
  --save_path /root/ocr_output

实验情况:无有效识别，原因可能在于python脚本将图片横向压缩太大，导致文字失真
结论:暂时没想到解决方法（似乎可以尝试将整个图片倾斜一定角度？但这样会导致其他原本未倾斜的文字倾斜，造成漏检。或者使用深度学习专门学习春联的特征，但这样模型鲁棒性太差了）



2026/10/24
21:36
第七次实验
尝试使用python脚本对sign_10图片进行预处理
实验情况：
效果良好，绝大部分倾斜文字被正确识别，但是出现了一定的扭曲，可能是模型的文本行方向分类True造成的，建议做一次对照实验
结论：第八次实验会将模型的文本行方向分类功能关闭，进行对比，python脚本已保存在本地工作目录中()


2026/10/24
21:50
第八次实验
进行对比实验
运行一下代码
paddleocr ocr \
  --input /mnt/c/Users/chenmuyang/Desktop/sign_10.jpg \
  --device gpu \
  --lang ch \
  --use_doc_orientation_classify False \
  --use_doc_unwarping False \
  --use_textline_orientation False \
  --save_path /root/ocr_output
实验情况：效果略差，建议开启去扭曲、文本行方向功能
结论：使用第七次实验工作流程，目前针对sign_10的问题已经得到解决，但是该python脚本的鲁棒性较差，仍需要改进
下一步优化方向：用“纸张区域内的墨水投影”代替“纸张长轴”，颜色掩膜只负责定位纸张区域； 在区域内部提取墨迹（如黑字 vs 橙纸，灰度/颜色对比），再对墨迹做掩膜内的投影剖面或梯度方向统计，求文字行方向，增强其鲁棒性

  