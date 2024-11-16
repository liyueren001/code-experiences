import os
from json import loads
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString


def jsonToXml(json_path, xml_path):  # 打开josn和xml文件夹，使用程序需要在指定位置新建一个xml文件夹。
    with open(json_path, 'r', encoding='UTF-8') as json_file:
        load_dict = loads(json_file.read())
    my_item_func = lambda x: 'Annotation'
    xml = dicttoxml(load_dict, custom_root='Annotations', item_func=my_item_func, attr_type=False)
    dom = parseString(xml)
    with open(xml_path, 'w', encoding='UTF-8') as xml_file:
        xml_file.write(dom.toprettyxml())


def json_to_xml(json_dir, xml_dir):  # 文件转换函数
    if (os.path.exists(xml_dir) == False):
        os.makedirs(xml_dir)
    dir = os.listdir(json_dir)
    for file in dir:
        file_list = file.split(".")
        if (file_list[-1] == 'json'):
            jsonToXml(os.path.join(json_dir, file), os.path.join(xml_dir, file_list[0] + '.xml'))
    print('==============')
    print('文件转换已完成')
    print('==============')


if __name__ == '__main__':
    json_path = "C:/Users/31064\Documents\Tencent Files/310648225\FileRecv\myf\myf\json"
    # 转换单个.json文件json_path = "E:/zhongnan_Data/open_fire_label/1.json"
    xml_path = "C:/Users/31064\Documents\Tencent Files/310648225\FileRecv\myf\myf/xml"
    # 保存单个.xml文件xml_path = "E:/zhongnan_Data/xml/1.xml"
    json_to_xml(json_path, xml_path)

