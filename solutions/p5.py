#初始化水位大小计数
low_sum = 0
ok_sum = 0
full_sum = 0
index = 1  #创建索引，用于记录
status = ""  #创建状态变量，用于记录水位状态
#接收水位大小列表
Water_level_list = list(map(int, input().split()))

#遍历水位列表，用于统计水位状态数量以及输出答案
for water_level in Water_level_list:
    if water_level < 30:
        low_sum += 1
        status = "low"
    elif water_level >= 30 and water_level <= 70:
        ok_sum += 1
        status = "ok"
    else:
        full_sum += 1
        status = "full"
    print(f"tower_{index}: {status}")
    index += 1  #索引加一
print(f"low={low_sum} ok={ok_sum} full={full_sum}")