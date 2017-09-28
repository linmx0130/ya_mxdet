#!/usr/bin/python3 
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

from xml.etree import ElementTree as ET

def parseBndbox(bndbox_node):
    xmin = int(bndbox_node.find('xmin').text)
    xmax = int(bndbox_node.find('xmax').text)
    ymin = int(bndbox_node.find('ymin').text)
    ymax = int(bndbox_node.find('ymax').text)
    return [xmin, ymin, xmax, ymax]


def parseFile(filename):
    tree = ET.parse(filename)    
    # parse filename
    filename_node = tree.find('filename')
    ret = {}
    ret['filename'] = filename_node.text
    # parse size
    size_node = tree.find('size')
    ret['width'] = int(size_node.find('width').text)
    ret['height'] = int(size_node.find('height').text)
    ret['depth'] = int(size_node.find('depth').text)

    object_nodes = tree.findall('object')
    ret['objects'] = []
    for item in object_nodes:
        obj = {}
        obj['name'] = item.find('name').text
        obj['truncated'] = int(item.find('truncated').text)
        obj['difficult'] = int(item.find('difficult').text)
        obj['bndbox'] = parseBndbox(item.find('bndbox'))
        ret['objects'].append(obj)
    return ret

