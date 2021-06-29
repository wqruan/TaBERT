from table_bert import Table, Column
from table_bert import TableBertModel
import numpy as np
import time
import os
import torch
torch.cuda.set_device(0)
model = TableBertModel.from_pretrained(
    '/disk/wqruan/tabert/tabert_base_k1/model.bin',
)

table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165']
    ]
).tokenize(model.tokenizer)

# To visualize table in an IPython notebook:
# display(table.to_data_frame(), detokenize=True)

context = 'show me countries ranked by GDP'
#context = 'show me the number of countries whose GDP are higher than 10,000,000'

# model takes batched, tokenized inputs
ticks = time.time()
context_encoding, column_encoding, info_dict = model.encode(
    contexts=[model.tokenizer.tokenize(context)],
    tables=[table]
)
print( time.time() -ticks)
context_mat = context_encoding.detach().numpy()
column_mat = column_encoding.detach().numpy()
# print(context_mat.shape)
# print(column_mat.shape)
# print(context_mat[0][0].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][0])*np.linalg.norm(column_mat[0][0])))
# print(context_mat[0][1].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][1])*np.linalg.norm(column_mat[0][0])))
# print(context_mat[0][2].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][2])*np.linalg.norm(column_mat[0][0])))
# print(context_mat[0][3].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][3])*np.linalg.norm(column_mat[0][0])))
# print(context_mat[0][4].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][4])*np.linalg.norm(column_mat[0][0])))
# print(context_mat[0][5].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][5])*np.linalg.norm(column_mat[0][0])))
# print(context_mat[0][6].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][6])*np.linalg.norm(column_mat[0][0])))
# print(np.linalg.norm(context_mat[0][0] - column_mat[0][0]))
# print(np.linalg.norm(context_mat[0][1] - column_mat[0][0]))
# print(np.linalg.norm(context_mat[0][2] - column_mat[0][0]))
# print(np.linalg.norm(context_mat[0][3] - column_mat[0][0]))
# print(np.linalg.norm(context_mat[0][4] - column_mat[0][0]))
# print(np.linalg.norm(context_mat[0][5] - column_mat[0][0]))
# print(np.linalg.norm(context_mat[0][6] - column_mat[0][0]))
# #print(context_mat[0][6].dot(column_mat[0][0])/( np.linalg.norm(context_mat[0][6])*np.linalg.norm(column_mat[0][0])))
# # print(context_mat[0][6].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][6])*np.linalg.norm(column_mat[0][1])))
# # print(context_mat[0][5].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][5])*np.linalg.norm(column_mat[0][1])))
# # print(context_mat[0][4].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][4])*np.linalg.norm(column_mat[0][1])))
# print("seperate")

# print(np.linalg.norm(context_mat[0][0] - column_mat[0][1]))
# print(np.linalg.norm(context_mat[0][1] - column_mat[0][1]))
# print(np.linalg.norm(context_mat[0][2] - column_mat[0][1]))
# print(np.linalg.norm(context_mat[0][3] - column_mat[0][1]))
# print(np.linalg.norm(context_mat[0][4] - column_mat[0][1]))
# print(np.linalg.norm(context_mat[0][5] - column_mat[0][1]))
# print(np.linalg.norm(context_mat[0][6] - column_mat[0][1]))
# print(context_mat[0][0].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][0])*np.linalg.norm(column_mat[0][1])))
# print(context_mat[0][1].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][1])*np.linalg.norm(column_mat[0][1])))
# print(context_mat[0][2].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][2])*np.linalg.norm(column_mat[0][1])))
# print(context_mat[0][3].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][3])*np.linalg.norm(column_mat[0][1])))
# print(context_mat[0][4].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][4])*np.linalg.norm(column_mat[0][1])))
# print(context_mat[0][5].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][5])*np.linalg.norm(column_mat[0][1])))
# print(context_mat[0][6].dot(column_mat[0][1])/( np.linalg.norm(context_mat[0][6])*np.linalg.norm(column_mat[0][1])))
#print(info_dict)/