import sys

text = sys.stdin.read()


lines = len(text.splitlines())  #统计行数
words = len(text.split())  #统计单词数
chars = len(text.replace('\n', ''))  #统计字符数

print(f"lines={lines} words={words} chars={chars}")