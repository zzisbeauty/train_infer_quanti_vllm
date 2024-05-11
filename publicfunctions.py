# 工具方法
import os, sys, time, requests, json, re, pickle, gc
import pandas as pd
import numpy as np
# from zhconv import convert
from tqdm import tqdm
import csv
# import jsonlines

# ##################################################### 正则相关

# 判断是否包含中文
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')


# content = '我爱北京天安门'
# match = zhPattern.search(content)
# chn = '有中文：%s' % (match.group(0),) if match else None


# 过滤表情
def clean(desstr, restr=''):
    try:
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
    return co.sub(restr, desstr)


# 去除一些数字与英文

# text = 'DC防寒羊羔绒裤'
# text = 'BreathDetox 草本肺部清洁喷雾'
# text = '23100906FA 益生菌猴头菇山药燕麦米稀'

def is_digit_string(s):  # 判断数字是纯数字
    return bool(re.match(r'^\d+$', s))


def matchTextGigit(source, target):
    match = re.search(r'[a-zA-Z0-9]+', source)
    if match:
        matchText = match.group()
        if is_digit_string(matchText):  # 纯数字不去除
            return source, target
        if len(matchText) < 6:  # 过短说明也不是乱七八糟的数据也不去除
            return source, target
        else:
            source = source.replace(matchText, '')
            target = target.replace(matchText, '')
        return source, target
    else:
        return source, target


# ##################################################### 数据库操作
# import pymysql


# # 读取数据库方法一
# # 参考 pandas 读取数据库

# # 读取数据库方法二
# def mysqlConnect(host, port, db, user, passwd):
#     conn = pymysql.connect(host=host, port=port, user=user, db=db, passwd=passwd, charset='utf8')
#     cursor = conn.cursor()
#     return conn, cursor


# def insserData(host, port, db, user, passwd, tabel, datas):
#     conn, cursor = mysqlConnect(host, port, db, user, passwd)
#     insertSql = """ insert into {} (lang_id, trans_source,trans_target, origin, translated) values (%s, %s, %s, %s, %s) """.format(
#         tabel)
#     cursor.executemany(insertSql, datas)
#     conn.commit()
#     print('save success')


# ##################################################### 路径相关

# 获得根路径
# 参考：https://blog.csdn.net/lovedingd/article/details/126479745
def getRootPath():
    # 获取文件目录
    curPath = os.path.abspath(os.path.dirname(__file__))
    # 获取项目根路径，内容为当前项目的名字
    rootPath = curPath[:curPath.find('项目名称') + len('项目名称')]
    return rootPath


# ##################################################### 时间操作
from datetime import datetime

now_time = str(datetime.now())[:10]  # 2024-01-01



# ##################################################### 日志操作
import logging

def logger_init(log_file_name='monitor', log_level=logging.INFO, log_dir='./logs/', only_file=False):
    saveLogsByDate = os.path.join(log_dir, str(datetime.now())[:10])
    os.makedirs(saveLogsByDate) if not os.path.exists(saveLogsByDate) else 1
    log_path = os.path.join(saveLogsByDate, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path, level=log_level,format=formatter,datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level, format=formatter, datefmt='%Y-%m-%d %H:%M:%S',
                            # filename=log_path, filemode='a',  # 设置这个就不能输出得到控制台
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]) # 即到控制台，又到文件