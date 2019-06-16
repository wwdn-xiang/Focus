#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py    
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-6-16 下午5:56   alpha      1.0         None
'''
# a = 3
# b = [1, 2]
# assert a in b

with open('/home/xxh/Focus/tf_tools/train.txt') as f:
    for line in f.readlines():
        print(line.strip())
