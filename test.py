# #coding=utf-8
# import pandas as pd
# excel_path1 = 'D:/conf_test/user_bk.xlsx'
# excel_path2 = 'D:/conf_test/userList.xlsx'
# d1 = pd.read_excel(excel_path1, sheetname='Sheet1')
# d2 = pd.read_excel(excel_path2, sheetname='userid&num&time')
# #print(d['Sheet1'].example_column_name)
# df1=d1[['user_id','create_date']]
# df2=d2[['user_id']]
#
# id_list = list(d2['user_id'])
#
#
#
# # print(df1)
# # print(df2)
# print('data load finished')
# dict_country = df1.set_index('user_id').T.to_dict('dict')
# print(dict_country)
# print(dict_country[4186794])
#
# result_dict = {}
#
# for uid in id_list:
#     result_dict[uid] = dict_country[uid]['create_date']
#
# print(result_dict)
#
# result_df = pd.DataFrame(list(result_dict.items()), columns=['user_id', 'create_date'])
# merge_df = pd.merge(d2, result_df, on='user_id')
#
# merge_df.to_csv('D:/merge_result.csv', index=None)
#
# #4186794: {'create_date': '2017-11-19 15:23:36'},
#
# #
# # print('开始')
# # gtDict=dict((key,value) for key,value in dict_country.items() if key in df2['user_id'].tolist())
# # print('执行ok')
# # framdic=pd.DataFrame(gtDict)
# # framdic.to_csv(r'D:\PythonProject\compute\data\user.csv')


# nums = [1,2,3,4,5,6,7,8,9]
#
# print(nums[-2:])

import tensorflow as tf

# x=tf.constant([[1,2],[1,2]])
# y=tf.constant([[1,1],[1,2]])
# z=tf.add(x,y)
#
# x1=tf.constant(1)
# y1=tf.constant(2)
# z1=tf.add(x1,y1)
#
# x2=tf.constant(2)
# y2=tf.constant([1,2])
# z2=tf.add(x2,y2)
#
# x3=tf.constant([[1,2],[1,2]])
# y3=tf.constant([[1,2]])
# z3=tf.add(x3,y3)
#
# x4 = tf.truncated_normal(shape=[1, 1], mean=0.0, stddev=1.0, dtype=tf.float32)
# y4 = tf.truncated_normal(shape=[1, 2], mean= 0.0, stddev=1.0, dtype=tf.float32)
# z4 = tf.add(x4, y4)
#
# with tf.Session() as sess:
#     z_result,z1_result,z2_result,z3_result, x4, y4, z4_result = sess.run([z,z1,z2,z3, x4, y4, z4])
#     print('z =\n%s'%(z_result))
#     print('z1 =%s'%(z1_result))
#     print('z2 =%s'%(z2_result))
#     print('z3 =%s'%(z3_result))
#
#     print('x4.shape = ', x4.shape)
#     print('y4.shape = ', y4.shape)
#
#     print(x4)
#     print(y4)
#     print(z4_result)

data = [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]]
gather_data = tf.gather(data, [0, 2])
with tf.Session() as sess:
    gather_data = sess.run(gather_data)
    print(gather_data)