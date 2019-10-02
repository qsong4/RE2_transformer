# import numpy as np
# from matplotlib import pyplot as plt
#
# x = [0, 1, 2, 3, 4, 5, 6]
# y = [86.5, 87.0, 87.6, 87.9, 88.6, 88.2, 88.1]
#
# plt.xlabel("number of dense layer")
# plt.ylabel("Acc of test set")
# plt.xlim(0, 8)
# plt.ylim(86, 89)
# plt.plot(x, y)
# plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # 生成画布的大小
# plt.grid(axis='y')
# plt.show()


# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# import matplotlib.ticker as ticker
#
# data = np.random.rand(10, 7)
#
# d = data
# d = d.transpose()
# col = ['a'] * 10    #需要显示的词
# index = ['b'] * 7  #需要显示的词
# df = pd.DataFrame(d, columns=col, index=index )
#
# fig = plt.figure(figsize=(8,8))
#
# for i in range(8):
#     ax = fig.add_subplot(241+i)
#
#     cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
#     #cax = ax.matshow(df)
#     fig.colorbar(cax)
#
#     tick_spacing = 1
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#
#     # fontdict = {'rotation': 'vertical'}    #设置文字旋转
#     fontdict = {'rotation': 90}       #或者这样设置文字旋转
#     #ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
#     # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
#     ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
#     ax.set_yticklabels([''] + list(df.index))
#
# plt.show()
# import matplotlib.pyplot as plt
# #plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
# plt.figure(figsize=(6,6))#将画布设定为正方形，则绘制的饼图是正圆
# label=['Frogs','Logs','Dogs', 'Hogs']#定义饼图的标签，标签是列表
# explode=[0.0,0.0,0.0,0.08]#设定各项距离圆心n个半径
# #plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
# values=[15,10,45,30]
# plt.pie(values,explode=explode,labels=label,autopct='%1.1f%%', shadow=True)#绘制饼图
# plt.show()

import numpy as np
# t2 = np.random.rand(10*10)
# t2.resize(10, 10)
# #最小值索引
# print(np.where(t2 == np.min(t2)))
# #最大值索引
# print(np.where(t2 == np.max(t2)))
# #print(t2)

# t1 = np.arange(start=0,stop=5,step=1,dtype=int)
# t1.resize(1,5)
# #print(t1)
# result = np.repeat(t1, 5, axis=0)
# print(result)

a = [1,2,3,4,5]
a = np.array(a)
res = []
res = np.array(res)
for i in range(len(a)):
    np.append(res, a[i])
    if i != len(a)-1:
        res.append('000')
print(res)

