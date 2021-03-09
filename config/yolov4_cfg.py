# coding=utf-8
# project
import os.path as osp
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

DATA_PATH = osp.join(PROJECT_PATH, 'data')

ATTENTION = {"TYPE": "NONE"}  # attention type:SEnet、CBAM or NONE