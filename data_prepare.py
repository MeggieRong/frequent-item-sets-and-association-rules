import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import os
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.dates as mdates

mpl.rcParams['font.sans-serif'] = ['SimHei']
import numpy as np
from collections import defaultdict


def data_prepare(columns, conn=None, filepath=None):
    column_list = [v for k, v in columns.items()]
    column_str = ','.join(column_list)
    rename_c = {v: k for k, v in columns.items()}
    base_data = pd.DataFrame()
    if conn:
        base_data = pd.read_sql(
            """select {} from ks_sale_order_detail where update_time > '2022-2-16'""".format(column_str), conn).rename(
            columns=rename_c)
    elif filepath:
        for (dirpath, dirnames, filenames) in os.walk(filepath):
            for name in filenames:
                if name.endswith('.xlsx'):
                    file_path = os.path.join(dirpath, name)
                    temp_data = pd.read_excel(file_path)[column_list].rename(columns=rename_c)
                    base_data = pd.concat([base_data, temp_data])

    return base_data


def data_process(project, df, sku_ratio):
    # 时间维度的考虑
    # order_with_time = df[['datetime', 'order_id']].drop_duplicates()
    # day_c = len(order_with_time['datetime'].unique())
    # df = df[['order_id', 'sku_code']].copy()
    # 合并原始订单
    base_data_g = df.groupby('order_id')['sku_code'].agg(set).reset_index()
    # 筛选出含复数SKU的订单
    base_data_g_f = base_data_g[base_data_g['sku_code'].apply(lambda x: len(x) > 1)].copy()
    # 根据百分比动态调整支持度
    top_ratio = sku_ratio
    # 筛选频繁元素
    sku = df['sku_code'].unique()
    supported_data = df.groupby('sku_code')['sku_code'].count().rename('sup_c').reset_index()
    supported_data['sup_r'] = supported_data['sup_c'] / len(base_data_g)
    # supported_data = supported_data[supported_data['sup_r'] >= min_sup]['sku_code'].tolist()
    # supported_data = supported_data[supported_data['sup_c'] >= min_count]['sku_code'].tolist()
    supported_data = supported_data.sort_values('sup_r', ascending=False)
    all_min_sup = supported_data['sup_r'].min()
    supported_data = supported_data.head(int(len(supported_data) * top_ratio))

    min_sup = supported_data['sup_r'].min()
    min_count = supported_data['sup_c'].min()
    adata = df[df['sku_code'].isin(supported_data['sku_code'])]  # a类基础数据
    data = adata.groupby('order_id')['sku_code'].agg(set).reset_index()  # a类订单总数
    # 筛选多项频繁项集
    data_c = data[data['sku_code'].apply(lambda x: len(x) > 1)]['sku_code'].tolist()  # a类多数订单总数
    # 实现onehot
    one_hot_l = [{j: True for j in i} for i in data_c]
    temp = pd.DataFrame(one_hot_l).fillna(False)

    # 手动调整最小支持度
    min_sup = 0.00000001
    # 自动调整最小支持度方案为目标数据集的30分位数 # 永辉不适用
    # 再次统计sku出现频次
    # s_c = defaultdict(int)
    # for s in data_c:
    #     for item in s:
    #         s_c[item] += 1
    # min_sup = pd.Series(s_c.values()).quantile(0.3)/len(data_c)

    # result1 = apriori(temp, min_support=min_sup, max_len=3, use_colnames=True)
    # result1_2 = result1[result1['itemsets'].apply(lambda x: len(x) == 2)].sort_values('support', ascending=False)
    result2 = fpgrowth(temp, min_support=min_sup, max_len=3, use_colnames=True)
    result2['order_count'] = result2['support'] * len(temp)
    r_ = result2[result2['itemsets'].apply(lambda x: len(x) > 1)]
    output_workbook = pd.ExcelWriter('{}关联关系总览.xlsx'.format(project))
    r_.to_excel(output_workbook, sheet_name='关联规则频次明细', encoding='utf-8', index=False)

    # result2_c = result2[result2['itemsets'].apply(lambda x: len(x) > 1)].sort_values('support', ascending=False)
    min_confidence = 0.8
    rule = association_rules(result2, metric='confidence', min_threshold=min_confidence)
    rule['conn'] = rule.apply(lambda x: x['antecedents'] | x['consequents'], axis=1)
    rule['order_count'] = rule['support'] * len(temp)
    r_sku = set([i for items in rule['conn'].values for i in items])

    def r_sku_count(sku_codes):
        r = r_sku & sku_codes
        if len(r) >= 2:
            return r
        else:
            return None

    base_data_g_f['r_sku'] = base_data_g_f['sku_code'].apply(r_sku_count)
    r_sku_order_df = base_data_g_f[~base_data_g_f['r_sku'].isna()]
    r_sku_order_df.to_excel(output_workbook, sheet_name='所有数据中包含的关联sku明细', encoding='utf-8', index=False)
    # r_sku_order = df[df['sku_code'].isin(r_sku)]['order_id'].unique()
    # r_sku_order_df = df[df['order_id'].isin(r_sku_order)].groupby('order_id')['sku_code'].agg(set).reset_index()
    # r_sku_order_df_f = r_sku_order_df[r_sku_order_df['sku_code'].apply(lambda x: len(x) > 1)]

    rule.to_excel(output_workbook, sheet_name='符合置信度的关联关系明细', encoding='utf-8', index=False)
    desc = {}
    desc['项目名称'] = project
    desc['样本总行数'] = len(df)
    desc['样本总数(订单数)'] = len(base_data_g)
    desc['行单比'] = desc['样本总行数'] / desc['样本总数(订单数)']
    desc['SKU总数'] = len(sku)
    desc['总最低支持度'] = all_min_sup
    desc['多项样本数(订单数)'] = len(base_data_g_f)
    desc['多项样本占比'] = desc['多项样本数(订单数)'] / desc['样本总数(订单数)']
    desc['ASKU筛选比例'] = top_ratio
    desc['ASKU数'] = len(supported_data)
    desc['ASKU最大支持度'] = supported_data['sup_r'].max()
    desc['ASKU最大出现次数'] = supported_data['sup_c'].max()
    desc['ASKU最小支持度'] = min_sup
    desc['ASKU最小出现次数'] = min_count
    desc['ASKU样本数'] = len(data)
    desc['ASKU多项样本数'] = len(data_c)
    # desc['ASKU2项频繁项数'] = len(result1_2)
    # desc['ASKU2项频繁项最高支持度'] = result1_2['support'].max()
    # desc['ASKU2项频繁项最低支持度'] = result1_2['support'].min()
    # desc['ASKU2项频繁项平均支持度'] = result1_2['support'].mean()
    desc['最小置信度'] = min_confidence
    desc['符合置信度的关联规则数'] = len(rule)
    desc['关联规则平均支持度'] = rule['support'].mean()
    desc['存在关联规则的SKU数'] = len(r_sku)
    desc['存在关联SKU的订单数'] = len(r_sku_order_df)
    desc['关联SKU订单占比'] = desc['存在关联SKU的订单数'] / desc['样本总数(订单数)']
    desc = pd.DataFrame(data=desc.items(), columns=['指标', '值'])
    desc.to_excel(output_workbook, sheet_name='总览', encoding='utf-8', index=False)
    output_workbook.save()
    return r_
    # pd.DataFrame([desc]).to_json('{}近3月.json'.format(project), force_ascii=False, orient='records')


def data_process_with_not_sku_filter(df, date):
    # 时间维度的考虑
    # order_with_time = df[['datetime', 'order_id']].drop_duplicates()
    # day_c = len(order_with_time['datetime'].unique())
    # df = df[['order_id', 'sku_code']].copy()
    # 合并原始订单
    base_data_g = df.groupby('order_id')['sku_code'].agg(set).reset_index()
    # 筛选出含复数SKU的订单
    base_data_g_f = base_data_g[base_data_g['sku_code'].apply(lambda x: len(x) > 1)]
    if base_data_g_f.empty:
        return None
    # # 根据百分比动态调整支持度
    # top_ratio = sku_ratio
    # # 筛选频繁元素
    # sku = df['sku_code'].unique()
    # supported_data = df.groupby('sku_code')['sku_code'].count().rename('sup_c').reset_index()
    # supported_data['sup_r'] = supported_data['sup_c'] / len(base_data_g)
    # # supported_data = supported_data[supported_data['sup_r'] >= min_sup]['sku_code'].tolist()
    # # supported_data = supported_data[supported_data['sup_c'] >= min_count]['sku_code'].tolist()
    # supported_data = supported_data.sort_values('sup_r', ascending=False)
    # all_min_sup = supported_data['sup_r'].min()
    # supported_data = supported_data.head(int(len(supported_data) * top_ratio))
    #
    # # min_sup = supported_data['sup_r'].min()
    # min_count = supported_data['sup_c'].min()
    # adata = df[df['sku_code'].isin(supported_data['sku_code'])]
    # data = adata.groupby('order_id')['sku_code'].agg(set).reset_index()
    # # 筛选多项频繁项集
    # data_c = data[data['sku_code'].apply(lambda x: len(x) > 1)]['sku_code'].tolist()
    # 实现onehot
    one_hot_l = [{j: True for j in i} for i in base_data_g_f['sku_code']]
    temp = pd.DataFrame(one_hot_l).fillna(False)

    # 手动调整最小支持度
    min_sup = 0.00000001

    # result1 = apriori(temp, min_support=min_sup, max_len=3, use_colnames=True)
    # result1_2 = result1[result1['itemsets'].apply(lambda x: len(x) == 2)].sort_values('support', ascending=False)
    result2 = fpgrowth(temp, min_support=min_sup, max_len=3, use_colnames=True)
    # result2['order_count'] = result2['support'] * len(base_data_g_f)
    # r_ = result2[(result2['itemsets'].apply(lambda x: len(x) > 1)) & (result2['order_count'] >= 5)].copy()
    # r_['date'] = date
    print(date)
    min_confidence = 0.00000001
    rule = association_rules(result2, metric='confidence', min_threshold=min_confidence)
    rule['order_count'] = rule['support'] * len(base_data_g_f)
    rule['date'] = date
    rule = rule[rule['order_count'] >= 5].copy()
    return rule.to_dict(orient='records')


def plot_all(project, df=None):
    # df = pd.read_excel('yhhotSku_频次总览(0.1).xlsx')
    listBins = [1, 6, 11, 16, 21, 10000]
    listLabels = ['1_5', '6_10', '11_15', '16_20', 'More than 20']
    #  按照规则订单数的分组 规则数的分布
    df['group'] = pd.cut(df['order_count'], bins=listBins, labels=listLabels, include_lowest=True, right=False)
    pic1_data = df.groupby(['group'])['order_count'].count().reset_index().rename(columns={'order_count': 'rule_count'})
    labels = pic1_data['group'].apply(lambda x: str(x))
    value = pic1_data['rule_count'].apply(lambda x: int(x))
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, value, width=0.35)
    ax.set_ylabel('rule count')
    ax.set_title('rule by order count')
    ax.set_xticks(x, labels)
    ax.set_xlabel('rule`s order count')
    ax.bar_label(rects1, padding=3)
    # fig.tight_layout()
    plt.savefig('{}按订单数分组的规则的计数.jpg'.format(project))
    # plt.show()
    # 按照规则订单数的分组 sku分布
    # pic2_data = df.groupby(['group'])['itemsets'].agg(set).reset_index()
    # pic2_data['sku_count'] = pic2_data['itemsets'].apply(lambda x: len(set([i for items in x for i in items])))
    # labels = pic2_data['group']
    # value = pic2_data['sku_count']
    # rects2 = ax.bar(x, value, width=0.35)
    # ax.set_ylabel('sku count')
    # ax.set_title('sku count group by rule`s order count')
    # ax.set_xticks(x, labels)
    # ax.set_xlabel('rule`s order count')
    # ax.bar_label(rects2, padding=3)
    # # fig.tight_layout()
    # plt.savefig('{}规则的sku数分布.jpg'.format(project))

    # 按照规则订单数的分组 订单数合计分布
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d km'))
    pic3_data = df.groupby(['group'])['order_count'].sum().reset_index()
    labels = pic3_data['group']
    value = pic3_data['order_count'] / 10000
    x = np.arange(len(labels))
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d w'))
    fig, ax = plt.subplots()
    rects3 = ax.bar(labels, value, width=0.35)
    ax.set_ylabel('order sum')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d w'))
    ax.set_title('order sum group by rule`s order count')
    ax.set_xticks(x, labels)
    ax.set_xlabel('rule`s order count')
    ax.bar_label(rects3, padding=3)
    # ax.legend(["单位：万"])
    # plt.show()
    # fig.tight_layout()
    plt.savefig('{}按订单数分组的规则的订单数合计.jpg'.format(project))
    print(1)


def plot_byday(project, df=None):
    # 每天所有规则涉及的的sku总数
    df_sku = df.groupby(['date'])['itemsets'].agg(set).reset_index()
    df_sku['sku_count'] = df_sku['itemsets'].apply(lambda x: len(set([i for items in x for i in items])))
    # 每天的规则数
    df_rule = df.groupby(['date'])['itemsets'].count().reset_index()
    # 每天的订单数
    df_order = df.drop_duplicates(['date'], keep='first').copy()
    df_order['order_count_all'] = df_order['order_count'] / df_order['support']
    df_order = df_order[['date', 'order_count_all']]
    df_rule = df.groupby(['date'])['itemsets'].count().reset_index()
    # 每天所有涉及规则的订单数的整体水平?
    df_order_sum = df.groupby(['date'])['order_count'].agg(sum).reset_index()  # 求和
    df_mode = df.groupby(['date'])['order_count'].agg(pd.Series.mode).reset_index()  # 众数
    df_quantile = df.groupby(['date'])['order_count'].quantile(0.5).reset_index()  # 分位数 0.5即为中位数
    df_var = df.groupby(['date'])['order_count'].var().reset_index()  # 方差
    pic1_data = pd.merge(df_sku, df_rule, on=['date'], how='left')
    pic1_data = pd.merge(pic1_data, df_order, on=['date'], how='left')
    # x = np.linspace(pic1_data['sku_count'])
    y1 = pic1_data['sku_count']
    y2 = pic1_data['order_count_all']
    y3 = pic1_data['itemsets_y']
    # 判断转化系数
    l_dict = {0: y1, 1: y2, 2: y3}
    index = 0
    max_v = 0
    for i, v in l_dict.items():
        if v.max() > max_v:
            index = i
    t = l_dict[index]
    max_len = len(str(int(t.mean())))
    for i, v in l_dict.items():
        if i == index:
            continue
        else:
            c_len = len(str(int(v.mean())))
            r = max_len - c_len
            if r:
                r = 10 ** r
                l_dict[i] = v * r
    fig, ax = plt.subplots()
    # locator = mdates.AutoDateLocator(minticks=2, maxticks=7)
    # Using set_dashes() to modify dashing of an existing line
    labels = pd.to_datetime(pic1_data['date'], format='%Y-%m-%d')
    months = mdates.MonthLocator()  # every month # 如果要定位年 years = mdates.YearLocator()   # every year
    ax.xaxis.set_major_locator(months)  # locator的功能是定位刻度，哪个地方要有个刻度
    month_format = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(month_format)  # formatter的功能是刻度上的标签以什么形式显示。
    # Using plot(..., dashes=...) to set the dashing when creating a line
    # ax.get_yaxis().set_visible(False)  # 隐藏y轴
    line1 = ax.plot(labels, l_dict[0], label='sku_count', dashes=[2])
    # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    line2 = ax.plot(labels, l_dict[1], dashes=[6, 2], label='order_count')
    line3 = ax.plot(labels, l_dict[2], dashes=[3, 0], label='rule_count')
    ax.legend()
    plt.savefig('{}日期趋势.jpg'.format(project))
    # plt.show()


if __name__ == '__main__':
    project = 'yh'
    # db_engine = 'mysql+pymysql'
    # user_name = 'dba2'
    # pwd = 'hairou_wangyitong_dba'
    # db_host = '10.1.204.3'
    # port = 3306
    # db_name = 'wms_core'
    # conn = create_engine(
    #     '{}://{}:{}@{}:{}/{}'.format(db_engine, user_name, pwd, db_host, port, db_name), encoding="utf-8",
    #     pool_recycle=3600, pool_pre_ping=True)
    # filepath = r"D:\workstation\data\京东订单数据"
    file = r'C:\Users\XXXX\Desktop\orderData\XXXX.xlsx'
    columns = {}
    # 京东字段['下发时间', '订单id', 'sku编码']
    # 万邑通字段 ['create_time','ks_sale_order_header_id','sku_code']
    # 永辉字段['update_time','ks_outbound_order_id', 'sku_code']
    columns['datetime'] = 'update_time'
    columns['order_id'] = 'ks_outbound_order_id'
    columns['sku_code'] = 'sku_code'
    # df = data_prepare(columns, conn=conn)
    df = pd.read_excel(file)
    df = df[df['update_time'] > '2022-02-18'][['ks_outbound_order_id', 'sku_code', 'update_time']].rename(
        columns={'ks_outbound_order_id': 'order_id', 'update_time': 'datetime'})
    df['date'] = df['datetime'].str[:10]
    date_list = df['date'].unique().tolist()
    # 总览
    # sku_ratio = 0.1
    # r_all = data_process(project, df[['order_id', 'sku_code']], sku_ratio)
    # plot_all(project, r_all)
    # 天明细
    r_day = []
    for date in date_list:
        r = data_process_with_not_sku_filter(df[df['date'] == date][['order_id', 'sku_code']], date)
        if r:
            r_day.extend(r)
    r_day = pd.DataFrame(r_day)
    r_day.to_excel('{}byday_rule明细.xlsx'.format(project), index=False)
    plot_byday(project, r_day)
