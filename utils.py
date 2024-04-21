import copy
import glob
import json
import os
import timeit
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from multiprocessing import Pool


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


def deep_dict(n):
    if n == 1:
        return lambda: defaultdict(set)
    return lambda: deep_dict(n - 1)


def auto_configure_device_map(num_gpus, device_map, num_trans_layers):
    per_gpu_layers = (num_trans_layers + 2) / num_gpus
    # from https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    if not device_map:
        device_map = {'transformer.embedding.word_embeddings': 0, 'transformer.encoder.final_layernorm': 0, 'transformer.output_layer': 0, 'transformer.rotary_pos_emb': 0, 'lm_head': 0}
        patten = 'transformer.encoder.layers'
    else:
        patten = 'model.layers'

    used, gpu_target = 2, 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'{patten}.{i}'] = gpu_target
        used += 1
    return device_map


def load_model_on_gpus(model, num_gpus=2, device_map=None, num_trans_layers=28):
    if num_gpus < 2 and device_map is None:
        return model.half()
    else:
        from accelerate import dispatch_model
        model = model.half()
        device_map = auto_configure_device_map(num_gpus, device_map, num_trans_layers)
        model = dispatch_model(model, device_map=device_map)
    return model


def check_model_para(model):
    print(model.state_dict)
    for name, para in model.named_parameters():
        print(f'{name:} {para.size()}', f'{para.requires_grad=}')


def zip_mask(a, b, mask):
    iter_a, iter_b = iter(a), iter(b)
    if callable(mask):
        # assert len(a) == (valid := sum(mask(i) for i in b)), f'have mismatch length with `{len(a)}`; `{valid}`'
        for i in b:
            yield next(iter_a) if mask(i) else None, next(iter_b)
    else:
        # assert (select := set(mask)) == {1, False}, f'bad input mask selections {select}'
        # assert len(b) == len(mask) > len(a), f'bad input shapes'
        # assert len(a) == (valid := sum(mask)), f'have mismatch length with `{len(a)}`; `{valid}`'
        for m in mask:
            yield next(iter_a) if m else None, next(iter_b)


def zh_normal(text):
    return punc_pattern.sub('', text)


def en_normal(text):
    return [s for i in text if (s := punc_pattern.sub('', i))]


def dedup_overlap(dic):
    return set(j for x, y in combinations((dic[i] for i in sorted(dic.keys())), 2) for i in x for j in y if i in j or '+' in j)


def print_chinese_flag():
    print(chr(127464) + chr(127475))


def print_self():
    with open(__file__, 'r') as file:
        code = file.read()
        print(code)


def update_stopwords():
    from zhconv import convert
    import pickle

    stopwords = pickle.load(open('bins/stopwords.bin', 'rb'))
    for i in ['我', '能', '管', '下', '上', '有', '会', '人', '除', '只', '不', '自', '到', '后', '用', '种', '内', '边', '看', '凡', '好', '他', '本', '同', '可', '打', '那', '你', '待', '论', '在',
              '才', '多', '对', '该', '最', '您', '大', '小', '宁', '别', '乘', '照', '经', '各', '赶', '顺', '起', '要', '过', '一', '己', '今', '离', '尽', '望', '任', '第', '依', '腾', '归', '无',
              '地', '欤', '嗳', '砰', '朝', '著', '临', '拿', '著', '靠', '据', '个']:
        stopwords['zh'].discard(i)
    # print([(i, convert(i, 'zh-tw')) for i in stopwords['zh']])
    stopwords['zh'].update([convert(i, 'zh-tw') for i in stopwords['zh']])
    stopwords['zh'].discard('著')
    db = {

        'programming_langs': {'assembly', 'batchfile', 'c-sharp', 'c', 'cmake', 'cpp', 'css', 'cuda', 'Dockerfile', 'fortran', 'go', 'haskell', 'html', 'java', 'javascript', 'julia', 'lua',
                              'Makefile', 'markdown', 'perl', 'php', 'powershell', 'python', 'ruby', 'rust', 'scala', 'shell', 'sql', 'swift', 'tex', 'typescript', 'visual-basic'},

        'extensions': {
            'image': {'.btif', '.mmr', '.rwl', '.bmp', '.g3', '.pic', '.dib', '.dwg', '.crw', '.fhc', '.png', '.sr2', '.ico', '.pict', '.rgb', '.wmf', '.mj2', '.nrw', '.tif', '.rw2', '.fh4', '.xpm',
                      '.srf', '.jfi', '.djvu', '.avifs', '.psd', '.pjpg', '.erf', '.fh', '.jpm', '.jpe', '.arw', '.jpx', '.cr2', '.pcx', '.fpx', '.dxf', '.mdi', '.jp2', '.j2k', '.pef', '.pbm',
                      '.tiff', '.pct', '.k25', '.ai', '.cur', '.ief', '.ptx', '.cgm', '.webp', '.apng', '.svgz', '.ind', '.ppm', '.mrw', '.raw', '.fst', '.jfif-tbnl', '.wbmp', '.kdc', '.xif', '.jpeg',
                      '.x3f', '.indt', '.xwd', '.pdf', '.heif', '.fbs', '.nef', '.indd', '.fh5', '.eps', '.pgm', '.cmx', '.jpf', '.gif', '.heic', '.pjp', '.xbm', '.raf', '.rlc', '.dng', '.ras',
                      '.npx', '.jif', '.djv', '.orf', '.pnm', '.fh7', '.svg', '.jpg', '.dcr', '.avif', '.icns', '.jfif'},
            'video': {'.asf', '.amv', '.ts', '.mpeg', '.M2TS', '.fli', '.h261', '.jpgm', '.avi', '.drc', '.mpa', '.wmx', '.mpe', '.fvt', '.rm', '.pyv', '.f4p', '.wm', '.f4v', '.yuv', '.mng', '.h263',
                      '.3gp', '.TS', '.ogg', '.mp4v', '.mj2', '.rmvb', '.viv', '.mp4', '.movie', '.wmv', '.h264', '.mpg4', '.f4b', '.flv', '.svi', '.f4a', '.asx', '.mpv', '.jpm', '.3g2', '.mpg',
                      '.vob', '.gif', '.mov', '.wvx', '.mkv', '.jpgv', '.m4v', '.m1v', '.mxf', '.qt', '.mp2', '.MTS', '.nsv', '.m4u', '.m2v', '.gifv', '.webm', '.mxu', '.mjp2', '.ogv', '.m4p',
                      '.roq'},
            'audio': {'.awb', '.m3a', '.3gpa', '.3gp2', '.nmf', '.aac', '.mka', '.rm', '.pya', '.m4b', '.opus', '.ogg', '.tta', '.m4r', '.vox', '.3g2', '.ecelp7470', '.ecelp4800', '.m2a', '.mmf',
                      '.mp2', '.m4a', '.ivs', '.mpc', '.midi', '.amr', '.dts', '.8svx', '.dss', '.mp4v', '.mp3', '.oga', '.rmp', '.movpkg', '.m3u', '.cda', '.mp4', '.rf64', '.aif', '.weba', '.3gp',
                      '.3gpp', '.wav', '.dvf', '.mogg', '.wv', '.aiff', '.rmi', '.aifc', '.ape', '.ra', '.act', '.eol', '.ram', '.kar', '.wma', '.aax', '.mpga', '.sln', '.adp', '.au', '.3ga', '.raw',
                      '.aff', '.flac', '.iklax', '.aacp', '.aa', '.voc', '.mid', '.msv', '.alac', '.m4v', '.spx', '.mp2a', '.wax', '.dtshd', '.webm', '.ecelp9600', '.gsm', '.lvp', '.snd', '.m4p',
                      '.3gpp2'},
            'chemical': {'.cxf', '.gcg', '.cmdf', '.sdf', '.cif', '.gjf', '.csm', '.gam', '.el', '.gjc', '.com', '.mmod', '.cml', '.csml', '.cpa', '.chm', '.ist', '.cer', '.kin', '.ihelm', '.xyz',
                         '.ds', '.c3d', '.istr', '.cef', '.fch', '.gen', '.xhelm', '.fchk', '.gamin', '.inchi', '.dx', '.ctx', '.gau', '.cdx', '.cbin', '.jsdraw', '.csf', '.helm', '.smiles', '.mol',
                         '.mmd', '.alc', '.jsd', '.jdx', '.emb', '.mcm', '.inp', '.bsd', '.cascii', '.smi', '.ctab', '.cub', '.embl', '.spc'}},

        'stopwords': stopwords

    }

    # print(stopwords['zh'])
    # text = '''  '''
    # print({i for i in text if i in stopwords['zh']})
    # print(len(db['stopwords']['zh']))

    pickle.dump(db, open('bins/storage.bin', 'wb'))


def syntax_sugar():
    *i, = range(4)
    print(i)

    i = 1234.56
    print(round(i, -3))


def dirichlet_kl():
    weights = []
    while len(weights) < 10:
        a = np.random.dirichlet(np.ones(10), size=1)
        b = np.random.dirichlet(np.ones(10), size=1)
        c = np.random.random_integers(1, 500, 10)

        ratio = np.log(a + 1e-8) - np.log(b + 1e-8)
        # assert a.sum() == 1. and b.sum() == 1. and c.sum() == 1.
        d = np.dot(c, np.transpose(ratio))
        if d[0] <= 0:
            continue
        weights.append(d)
    d = np.concatenate(weights)
    print(d)
    print(np.random.gumbel(size=d.shape))


def draw_single(data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    ITEMS = sum(data['count'])

    np.random.seed(42)

    # show signal minus `-`
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 400

    fg = sns.barplot(data, y='name', x='count')

    for p in fg.patches:
        percentage = p.get_width() / ITEMS * 100
        percentage = round(percentage, 2)

        x = p.get_x() + p.get_width() / 2
        y = p.get_y() + p.get_height() * 7 / 12

        fg.text(x, y, f'{percentage}%', color='black', fontsize=5)
        fg.text(p.get_x() + p.get_width(), p.get_y() + p.get_height(), f'{int(p.get_width())}', color='black', fontsize=5)
    fg.figure.set_size_inches(12, 6)

    plt.ylabel(f'domain')
    plt.xlabel(f'count')
    plt.tight_layout()
    # plt.savefig(f'plots/{METHOD}.png')
    plt.show()


def draw(input):
    import seaborn as sns
    import matplotlib.pyplot as plt
    np.random.seed(42)

    # show signal minus `-`
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 400

    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    for i, (data, ITEMS, y) in enumerate(input):
        fg = sns.barplot(data, y='name', x='count', axes=axes[i])
        axes[i].set_title(y)
        for p in fg.patches:
            percentage = p.get_width() / ITEMS * 100
            percentage = round(percentage, 2)

            x = p.get_x() + p.get_width() / 3
            y = p.get_y() + p.get_height() * 2 / 3

            fg.text(x, y, f'{percentage}%', color='black', fontsize=3)
            fg.text(p.get_x() + p.get_width(), p.get_y() + p.get_height() * 2 / 3, f'{int(p.get_width()):,}', color='black', fontsize=3)
        fg.figure.set_size_inches(15, 15)

        plt.ylabel(f'domain')
        plt.xlabel(f'count / {ITEMS:,}')
    plt.tight_layout()
    # plt.savefig(f'all_count_01.png')
    plt.show()


def draw_h(data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    def display_percent(figure):
        for ax in figure.axes.ravel():
            # add annotations
            for c in ax.containers:
                # custom label calculates percent and add an empty string so 0 value bars don't have a number
                labels = [f'{w:.1f}%' if (w := (v.get_height() / (len(data) / 100))) > 0 and w >= 2 else '' for v in c]
                ax.bar_label(c, labels=labels, label_type='edge', fontsize=4, rotation=90, padding=2, color=c[0].get_facecolor())
            ax.margins(y=0.1)

    np.random.seed(42)

    # show signal minus `-`
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 400
    # plt.rcParams['figure.figsize'] = (2000, 1500)
    # sns.set_theme(rc={'figure.dpi': 300})

    fg = sns.displot(data, stat='count', kde=True, multiple='layer')
    display_percent(fg)
    fg.figure.set_size_inches(12, 6)
    sns.move_legend(fg, loc='upper right', title='Distance', frameon=False)

    # plt.legend(fg.legend, title='Distance')
    plt.ylabel(f'Count / {len(data)}')
    plt.xlabel(f'similarity')
    plt.tight_layout()
    plt.show()


def load_jsonl_path(pattern):
    for file in glob.glob(pattern):
        with open(file) as f:
            for line in f:
                yield json.loads(line)


def tmp():
    _ = ['a'* 10 for _ in range(10000)]


colorful = Colorful()


if __name__ == '__main__':
    # from pdfminer.high_level import extract_text
    #
    # text = extract_text('../Downloads/pdf/2023-05_b385f696bae812fba4bafe651a6c5ff1.pdf')
    # print(text)

    # import docx
    # doc = docx.Document('../Downloads/Skywork13B.docx')
    # for i in doc.paragraphs:
    #     print(i.text)  # exit()

    # update_stopwords()

    # with open('extract/xml/normal/2023-05.pkl', 'rb') as f:
    #     end = os.fstat(f.fileno()).st_size
    #     while f.tell() != end:
    #         bs = pickle.load(f)


    import zlib
    import sys
    from datasketch import MinHash

    import psutil
    tmp()
    # print(f'utils main {os.getpid()=}')
    exit()

    from minhash import byteswap

    m = MinHash()
    m.update('怒舒适sfdsdfdf'.encode('utf-8'))
    a = byteswap(m.hashvalues)

    print(len(zlib.compress(a)))
    exit()

    b = {bytes(f'{i:01024d}', 'utf-8'): i for i in range(1000)}
    print(a.format('>'))
    print(2799269512 / 1024 ** 3)
