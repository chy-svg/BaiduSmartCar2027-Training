import sys

params = {}  #创建字典，用于接收键值对并排序

text = sys.stdin.read()  #接收文本

lines = text.splitlines()  #以行为单位接收

for line in lines:
    if not line.strip():  #如果该行为空行，则跳过 
        continue
    if line.startswith('#'): 
        continue  #如果该行是注释行，则跳过

    line_parts = line.split('=', 1)  #将该行按 = 分割为单词列表
    params[line_parts[0]] = line_parts[1]  #按键值对添加进字典  

sorted_items = sorted(params.items())  #对字典按 key 进行排序

for key, value in sorted_items:  #输出
    print(f"{key}: {value}")