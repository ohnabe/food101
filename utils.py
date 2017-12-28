# -*- coding: utf-8 -*-
"""
複数のモジュールで使用されるユーティリティ
"""

from chainer.cuda import get_device_from_id

def parse_kwargs(args):
    """キーワード引数文字列のパーズ

    ``"key1=value1,key2=value2,..."`` 形式の文字列をディクショナリ形式::

        { "key1": "value", "key2": "value", ... }

    に変換して返す.

    Args:
        args (str): キーワード引数文字列

    Returns:
        dict. 変換後のディクショナリ
    """

    if args == '':
        return {}

    kwargs = {}
    for arg in args.split(','):
        key, value = arg.split('=')
        kwargs[key] = value

    return kwargs


def setup_devices(ids):
    """GPU のセットアップ

    Args:
        ids (str):  GPU指定文字列

    Returns:
        dict.   (デバイス名, デバイスID)からなるディクショナリ
    """
    if ids == '':
        return {'main': -1}

    devices = parse_kwargs(ids)
    for key in devices:
        devices[key] = int(devices[key])

    get_device_from_id(devices['main']).use()
    return devices