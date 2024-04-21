import timeit
import numpy as np


def polynomial(x, coefficients, order):
    assert 0 < order == len(coefficients) - 1, colorful.red(f'unmatch order: {order} with coefficients: {len(coefficients)}')
    return sum(coefficients[i] * x ** o for i, o in enumerate(range(order, -1, -1)))


def now():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


def bar(iterator, title='bar'):
    from alive_progress import alive_it
    """ wrap the iterator as a progress-bar """
    return alive_it(iterator, title=title, bar='bubbles', spinner='horizontal')


def strike(text):
    """ 为字符串的每个字符添加删除线 `strike('hello world')` -> '̶h̶e̶l̶l̶o̶ ̶w̶o̶r̶l̶d' """
    return ''.join('\u0336{}'.format(c) for c in text)


def bytes2gigabytes(x):
    """ Converting Bytes to Megabytes """
    return x / 2 ** 30


def cosine_sim(a, b):
    return b.dot(a) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def msb(a):
    while p := a & (a - 1):
        a = p
    return a


class Colorful:
    """ Align_map: '<': 'left', '>': 'right', '^': 'mid' \n
        Mode map: {'0': 'Normal', '1': 'Bold/Highlight', '7': 'Reversed'} \n
        Front color map: {30: 'Black', 32: 'Green', 34: 'Blue', 36: 'Cyan', 40: 'White'} \n
        Back color map: {';40': 'Black', ';42': 'Green', ..., ';47': 'White'}  set `back_color` with '' to be default \n
        Font map: {7: 'normal', 1: 'bold' ...} """

    def __call__(self, content, mode='7', front_color=40, back_color=';40', align='^', width=30, unit='gb', rounding=2):
        aligned = '\033[{}{' + f':{align + str(width)}' + '}\033[0m'
        if type(content) is float:
            rounded = '{' + f':.{rounding}' + 'f}'
            return aligned.format(f'{mode};{front_color}{back_color}m', rounded.format(content) + unit)
        return aligned.format(f'{mode};{front_color}{back_color}m', content)

    @staticmethod
    def is_colorful(content):
        return content.startswith('\033') and content.endswith('\033[0m')

    @staticmethod
    def is_colorless(content):
        return not (content.startswith('\033') and content.endswith('\033[0m'))

    def red(self, content):
        return self(content, mode='1', front_color=31, back_color='', align='<', width=0)

    def green(self, content):
        return self(content, mode='1', front_color=32, back_color='', align='<', width=0)

    def yellow(self, content):
        return self(content, mode='1', front_color=33, back_color='', align='<', width=0)

    def blue(self, content):
        return self(content, mode='1', front_color=34, back_color='', align='<', width=0)

    def white(self, content):
        return self(content, mode='1', front_color=37, back_color='', align='<', width=0)

    def cbar(self, number):
        assert type(number) is int or float, 'should call by a `number` like'
        oct_width = len(str(number))
        width = 2 + oct_width + (oct_width - 1) // 3
        front_color = 32 if oct_width > 9 else 33 if oct_width > 6 else 31
        return self(f'{number:,d}', mode='1;7', rounding=0, front_color=front_color, width=width)

    def timer(self, start, end=timeit.default_timer()):
        hours, seconds = divmod(end - start, 3600)
        setting = {'mode': '1;7', 'rounding': 0, 'front_color': 36}
        hour_minute = self(hours, width=15, align='>', unit='h ', **setting)
        hour_minute += self(seconds / 60, width=15, align='<', unit='min', **setting)
        return self('Time consume: ', width=47, **setting) + hour_minute


colorful = Colorful()
