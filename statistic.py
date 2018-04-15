#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs

from dicttoxml import dicttoxml


def parse(json_file):
    with open(json_file, 'r') as f:
        json_str = f.read()
        json_object = json.loads(json_str)
        return json_object


if __name__ == "__main__":
    json_object = parse('../iNaturalist_image_2017/val_2017_bboxes.json')
    with codecs.open('inat_2017_label_map.pbtxt', 'w', 'utf-8') as f:
        for category in json_object['categories']:
            f.write(u'item {\n')
            # label map ID starts at 1
            f.write(u'  id: {0}\n'.format(int(category['id']) + 1))
            f.write(u"  name: '{0}'\n".format(category['name']))
            f.write(u'}\n')

    print(json_object)
