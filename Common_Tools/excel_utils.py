# encodeing:utf-8
import csv
import xlrd
import xlutils
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.colour_index = 4
    font.height = height
    style.font = font
    return style


'''
# 打开excel表
excel_obj = open_workbook('./pathology2019-11-26.xls')
# 获取excel表文件中各个子表的名称列表，返回一个名称的列表，可以通过子表的名称来遍历
sub_excel_names = excel_obj.sheet_names()
# 根据索引获取子表对象
sub_excel_obj = excel_obj.sheet_by_index(0)

# 根据索引获取对应的行，返回的是该行数据的列表
sub_excel_row0_value = sub_excel_obj.row_values(0)

# 根据内容，获取其对应的索引
index = sub_excel_row0_value.index('value')

# 根据索引获取对应的列,返回的是该列数据的列表
sub_excel_col0_value = sub_excel_obj.col_values(0)

# 根据内容，获取其对应的索引
index = sub_excel_col0_value.index('value')

# 新建一个空的excel表
f = xlwt.Workbook()
# 根据读取子表的名称,根据名称新建一个同名的子表并加入到新建的空excel表中
sheet = f.add_sheet('test.xls', cell_overwrite_ok=True)

# 遍历一行的内容
rows = 0
for i, content in enumerate(sub_excel_row0_value):
    print(content)
    sheet.write(rows, i, content)  # 参数分别为行索引，列索引，内容
rows += 1  # 写完一行，行数加一
# f.save("test.xls")
'''

'''
********************************注意**************************************
对于表格的修改，只能通过拷贝一份原始数据的方式，对拷贝数据作修改。
********************************注意**************************************
'''

if __name__ == '__main__':
    excel_m_obj = open_workbook('./pathology2019-11-26.xls')
    excel_z_obj = open_workbook('./Book1.xlsx')
    # 拷贝一份表格
    copy_excle_m = copy(excel_m_obj)
    # 获取拷贝表格的第一个子表
    newWs = copy_excle_m.get_sheet(0)

    m_sub_excel_obj = excel_m_obj.sheet_by_index(0)
    z_sub_excel_obj = excel_z_obj.sheet_by_index(0)

    z_values_clo_0 = z_sub_excel_obj.col_values(0)
    m_values_clo_0 = m_sub_excel_obj.col_values(0)

    for i, value in enumerate(m_values_clo_0):
        if i > 0:
            if value in z_values_clo_0:
                index = z_values_clo_0.index(value)

                z_row_values_index = z_sub_excel_obj.row_values(index)
                insert_value = z_row_values_index[1]
                # 在对应的行，列单元格内写入数据
                newWs.write(index, 2, insert_value)
    # 保存单元格
    copy_excle_m.save("debug.xls")
