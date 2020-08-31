import  pandas as pd
import os
import numpy as np
import datetime as dt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import model_selection

def  startread():
    dataname = ['acc_08_final.csv', 'acc_09_final.csv', 'acc_10_final.csv', 'acc_11_final.csv']
    to_path = "savefile/h.csv"
    for i in range(len(dataname)):
        path = dataname[i]
        data = pd.read_table(path, chunksize=500000, sep=',',
                             usecols=['CARD_ID', 'TRADE_TYPE', 'TRADE_ADDRESS', 'TRADE_DATE', 'OPERATOR'],
                             low_memory=False)
        for chunk in data:
            # print(chunk)
            # 删除工作人员的数据
            chunk = chunk[(chunk['OPERATOR'].isin([0]))]
            chunk = chunk.drop(['OPERATOR'], axis=1)
            # 人的ID
            # 日期处理
            DATE = pd.to_datetime(chunk.TRADE_DATE, format="%Y-%m-%d-%H.%M.%S.%f")
            DATE = DATE.apply(lambda x: datetime.strftime(x, "%Y-%m-%d"))
            chunk['TRADE_DATE'] = pd.to_datetime(DATE)  # 将数据类型转换为日期类型
            # 人的ID
            chunk = chunk.drop_duplicates(subset=['CARD_ID', 'TRADE_DATE'], keep='first')
            chunk = chunk.drop(['CARD_ID'], axis=1)
            # csv_data = csv_data.set_index('TRADE_DATE')#将日期作为索引
            chunk = chunk.groupby(['TRADE_DATE', 'TRADE_ADDRESS']).count().reset_index()
            # print(csv_data)
            chunk.to_csv(to_path, mode='a', index=False, header=False)
            # chunk.to_csv(csv_path)

def  AD_WK():
    """
    提取出每个站点的星期几的数据
    :return:
    """
    # print(data_f)
    for  i in range(da_al):
        d=da_a[i]#每次的站点
        daad=data_f[(data_f['TRADE_ADDRESS'].isin([d]))]
        # print(len(daad))
        daad_a=daad[['TRADE_DATE','TRAFFIC']]
        daad_d = pd.to_datetime(daad_a.TRADE_DATE, format="%Y-%m-%d")
        daad_d= daad_d.apply(lambda x: datetime.strftime(x, "%m%d"))
        daad_d=np.array(daad_d)
        arr = list(map(int, daad_d))
        print(arr)
        daad_a['TRADE_DATE']= arr
        # daad_a.loc[daad_a['TRADE_DATE']]= arr
        daad_dl=len(arr)
        for q in range(H):
            for p in range(R):
                print("星期{}".format(WE[p]))
                for k in range(daad_dl):

                 if arr[k] in WEEK[q][p]:
                     w=daad_a[daad_a['TRADE_DATE'].isin([arr[k]])]
                     data_path = os.path.join('savepart', '{0}站点星期{1}数据.csv'.format(d, WE[p]))
                     w.to_csv(data_path, mode='a', index=False, header=False)
def Pre_File():
    for i in range(da_al):
        # daad_a['TRADE_DATE']= arr
        # print(d)
        # daad_dl= len(arr)

        for p in range(R):
            # for k in range(daad_dl):
            print("预测结果{}星期{}".format(da_a[i], WE[p]))
            d = pd.read_csv('savepart/'+str(da_a[i])+'站点星期'+str(WE[p])+'数据.csv', header=None, names=['TRADE_DATE', 'TRAFFIC'])
            d['TRADE_DATE']=WH[p]
            # print(d['TRADE_DATE'])
            z=np.array(p+1).reshape(-1,1)
            X=d['TRADE_DATE']
            X = np.array(X).reshape(-1,1)
            y = d['TRAFFIC']
            y = np.array(y)
            # print(y)
            # 生成scikit-learn的线性回归对象
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
            clf = LinearRegression()
            # 开始训练
            clf.fit(X_train, y_train)
            # # 进行预测
            predict = clf.predict(z)
            perfile.append(int(predict))
            TRADE_ADDRESS.append(da_a[i])
            TRADE_DATE.append(WF[p])
            # data_path = os.path.join('sf', '{0}站点星期{1}数据.csv'.format(d, WE[q][p]))
        # print(TRADE_DATE)
    dada=pd.DataFrame(perfile,columns=['客流量(刷卡类型21+22)'])
    dada['TRADE_ADDRESS']=TRADE_ADDRESS
    dada['日期'] =TRADE_DATE
    dt.datetime.now().strftime('%m{m}%d{d}').format(m='月', d='日')
    # print(dada)
    # array = time.strptime(publish_Time, u"%Y年%m月%d日")
    # dt = datetime.strptime([].join(TRADE_DATE), '%m月%d日')
    dada = dada.sort_values(by=['TRADE_ADDRESS','日期'])
    dada=dada[['日期','TRADE_ADDRESS','客流量(刷卡类型21+22)']]
    # print(dada)
    dada.to_excel("sf/predict3.xlsx")

if __name__ == '__main__':
    startread()
    if os.path.exists('savefile/h.csv'):
        data_f = pd.read_csv('savefile/h.csv', header=None, names=['TRADE_DATE', 'TRADE_ADDRESS', 'TRAFFIC'])
        data_f = data_f.set_index('TRADE_DATE')
        data_f = data_f.groupby(['TRADE_ADDRESS', 'TRADE_DATE']).sum().reset_index()
        da_a = data_f['TRADE_ADDRESS'].unique()
        # print(data_f)
        da_al = len(da_a)
        da_d = data_f['TRADE_DATE'].unique()
        da_dl = len(da_d)
        WEEK1 = [[803, 810, 817, 824, 831], [804, 811, 818, 825], [805, 812, 819, 826], [806, 813, 820, 827], [807, 814, 821, 828],
                 [801, 808,815, 822, 829], [802, 809, 816, 823, 830]]
        WEEK2 = [[907, 914, 921, 928], [901, 908, 915, 922, 929], [902, 909, 916, 923, 930], [903, 910, 917, 924], [904, 911, 918, 925],
                 [905, 912, 919, 926], [906, 913, 920, 927]]
        WEEK3 = [[1005, 1012, 1019, 1026], [1006, 1013, 1020, 1027], [1007, 1014, 1021, 1028], [1001, 1008, 1015, 1022, 1029], [1002, 109, 1016, 1023, 1030],
                 [1003, 1010, 1017, 1024, 1031], [1004, 1011, 1018, 1025]]
        WEEK4 = [[1102, 1109, 1116, 1123, 1130], [1103, 1110, 1117, 1124], [1104, 1111, 1118, 1125], [1105, 1112, 1119, 1126], [1106, 1113, 1120, 1127],
                 [1107, 1114, 1121, 1128], [1101, 1108, 1115, 1122, 1129]]
        WEEK = [WEEK1, WEEK2, WEEK3, WEEK4]
        WE = ["一", "二", "三", "四", "五", "六", "日"]
        WH = [1,2,3,4,5,6,7]
        WF = ["12月28日", "12月29日", "12月30日", "12月31日", "12月25日", "12月26日", "12月27日"]
        H = len(WEEK)
        R = len(WEEK1)
        X_lately = [1, 2, 3, 4, 5, 6, 7]
        perfile = []
        TRADE_ADDRESS = []
        TRADE_DATE = []
        # print(type(WEEK1[1][1]))
        AD_WK()
        Pre_File()
