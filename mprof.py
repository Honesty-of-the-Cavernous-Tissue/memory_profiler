import os
import sys
import re
import copy
import time
import glob
import math
import logging
import itertools
import os.path as osp

import numpy as np
from cycler import cycler
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter

import memory_profiler as mp

logger = logging.getLogger(__name__)
logging.basicConfig()
color = mp.Colorful()


def get_profile_filenames(args):
    """Return list of profile filenames.

    Parameters
    ==========
    args (list)
        list of filename or integer. An integer is the index of the
        profile in the list of existing profiles. 0 is the oldest,
        -1 in the more recent.
        Non-existing files cause a ValueError exception to be thrown.

    Returns
    =======
    filenames (list)
        list of existing memory profile filenames. It is guaranteed
        that an given file name will not appear twice in this list.
    """
    profiles = glob.glob('mprofile_??????????????.dat')
    profiles.sort()

    if args == 'all':
        filenames = copy.copy(profiles)
    else:
        filenames = []
        for arg in args:
            if arg == '--':  # workaround
                continue
            try:
                index = int(arg)
            except ValueError:
                index = None
            if index is not None:
                try:
                    filename = profiles[index]
                except IndexError:
                    raise ValueError('Invalid index (non-existing file): %s' % arg)

                if filename not in filenames:
                    filenames.append(filename)
            else:
                if osp.isfile(arg):
                    if arg not in filenames:
                        filenames.append(arg)
                elif osp.isdir(arg):
                    raise ValueError('Path %s is a directory' % arg)
                else:
                    raise ValueError('File %s not found' % arg)

    # Add timestamp files, if any
    for filename in reversed(filenames):
        parts = osp.splitext(filename)
        timestamp_file = parts[0] + '_ts' + parts[1]
        if osp.isfile(timestamp_file) and timestamp_file not in filenames:
            filenames.append(timestamp_file)

    return filenames


def get_cmd_line(args):
    """ Given a set or arguments, compute command-line. """
    blanks = {' ', '\t'}
    return ' '.join(s if not blanks & set(s) else ' + s + ' for s in args)


def find_first_process(name):
    for i in mp.psutil.process_iter():
        if name in i.name():
            return i


def add_brackets(xloc, yloc, xshift=0, color='r', label=None, options=None):
    """Add two brackets on the memory line plot.

    This function uses the current figure.

    Parameters
    ==========
    xloc: tuple with 2 values
        brackets location (on horizontal axis).
    yloc: tuple with 2 values
        brackets location (on vertical axis)
    xshift: float
        value to subtract to xloc.
    """
    try:
        import pylab as pl
    except ImportError as e:
        print('matplotlib is needed for plotting.')
        print(e)
        sys.exit(1)
    height_ratio = 20.
    vsize = (pl.ylim()[1] - pl.ylim()[0]) / height_ratio
    hsize = (pl.xlim()[1] - pl.xlim()[0]) / (3. * height_ratio)

    bracket_x = pl.asarray([hsize, 0, 0, hsize])
    bracket_y = pl.asarray([vsize, vsize, -vsize, -vsize])

    # Matplotlib workaround: labels starting with _ aren't displayed
    if label[0] == '_':
        label = ' ' + label
    if options.xlim is None or options.xlim[0] <= (xloc[0] - xshift) <= options.xlim[1]:
        pl.plot(bracket_x + xloc[0] - xshift, bracket_y + yloc[0], '-' + color, linewidth=2, label=label)
    if options.xlim is None or options.xlim[0] <= (xloc[1] - xshift) <= options.xlim[1]:
        pl.plot(-bracket_x + xloc[1] - xshift, bracket_y + yloc[1], '-' + color, linewidth=2)

        # TODO: use matplotlib.patches.Polygon to draw a colored background for  # each function.

        # with maplotlib 1.2, use matplotlib.path.Path to create proper markers
        # see http://matplotlib.org/examples/pylab_examples/marker_path.html
        # This works with matplotlib 0.99.1  ## pl.plot(xloc[0], yloc[0], '<'+color, markersize=7, label=label)  ## pl.plot(xloc[1], yloc[1], '>'+color, markersize=7)


def read_mprofile_file(filename):
    """
    Read a mprofile file and return its content.
    Returns:
        content: dict
        Keys:
        - 'mem_usage': (list) memory usage values, in MiB
        - 'timestamp': (list) time instant for each memory usage value, in second
        - 'func_timestamp': (dict) for each function, timestamps and memory usage upon entering and exiting.
        - 'cmd_line': (str) command-line ran for this profile.
    """

    func_ts, cmd_line = {}, None
    mem_usage, timestamp = [], []
    children = defaultdict(list)
    with open(filename) as f:
        for line in f:
            if line == '\n':
                raise ValueError(color.red('Sampling time was too short'))
            field, value = line.split(maxsplit=1)
            if field == 'MEM':
                mem, ts = value.split()
                timestamp.append(float(ts))
                mem_usage.append(float(mem))

            elif field == 'FUNC':
                values = value.split()
                f_name, mem_start, start, mem_end, end = values[:5]
                ts = func_ts.get(f_name, [])
                to_append = [float(start), float(end), float(mem_start), float(mem_end)]
                if len(values) >= 6:
                    # There is a stack level field
                    stack_level = values[5]
                    to_append.append(int(stack_level))
                ts.append(to_append)
                func_ts[f_name] = ts

            elif field == 'CHLD':
                child_id, mem, ts = value.split()
                children[child_id].append((float(mem), float(ts)))

            elif field == 'CMDLINE':
                cmd_line = value
            else:
                pass

    return {'mem_usage': mem_usage, 'timestamp': timestamp, 'func_timestamp': func_ts, 'filename': filename, 'cmd_line': cmd_line, 'children': children}


FLAME_PLOTTER_VARS = {'hovered_rect': None, 'hovered_text': None, 'alpha': None}


def flame_plotter(filename, index=0, timestamps=True, children=True, options=None):
    try:
        import pylab as pl
    except ImportError as e:
        print('matplotlib is needed for plotting.')
        print(e)
        sys.exit(1)
    import numpy as np  # pylab requires numpy anyway
    mprofile = read_mprofile_file(filename)

    if len(mprofile['timestamp']) == 0:
        print('** No memory usage values have been found in the profile '
              'file.**\nFile path: {0}\n'
              'File may be empty or invalid.\n'
              'It can be deleted with "mprof rm {0}"'.format(mprofile['filename']))
        sys.exit(0)

    # Merge function timestamps and memory usage together
    ts = mprofile['func_timestamp']
    t = mprofile['timestamp']
    mem = mprofile['mem_usage']
    chld = mprofile['children']

    if len(ts) > 0:
        for values in ts.values():
            for v in values:
                t.extend(v[:2])
                mem.extend(v[2:4])

    mem = np.asarray(mem)
    t = np.asarray(t)
    ind = t.argsort()
    mem = mem[ind]
    t = t[ind]

    if ts:
        stack_size = 1 + max(ex[4] for executions in ts.values() for ex in executions)
    else:
        stack_size = 0

    def level_to_saturation(level):
        return 1 - 0.75 * level / stack_size

    colors = [itertools.cycle([pl.matplotlib.colors.hsv_to_rgb((0, level_to_saturation(level), 1)), pl.matplotlib.colors.hsv_to_rgb((0.1, level_to_saturation(level), 1)), ]) for level in
              range(stack_size)]

    # Plot curves
    global_start = float(t[0])
    t = t - global_start

    max_mem = mem.max()
    max_mem_ind = mem.argmax()

    # cmap = pl.cm.get_cmap('gist_rainbow')
    mem_line_colors = ('k', 'b', 'r', 'g', 'c', 'y', 'm')
    mem_line_label = time.strftime('%d / %m / %Y - start at %H:%M:%S', time.localtime(global_start)) + '.{0:03d}'.format(int(round(math.modf(global_start)[0] * 1000)))

    pl.plot(t, mem, '-' + mem_line_colors[index % len(mem_line_colors)], label=mem_line_label)

    bottom, top = pl.ylim()
    bottom += 0.001
    top -= 0.001

    ax = pl.gca()
    ax.grid(True)
    timestamp_ax = ax.twinx()
    timestamp_ax.set_yticks([])
    timestamp_ax.set_ylim((0, stack_size + 1))
    timestamp_ax.grid(False)

    # plot children, if any
    if len(chld) > 0 and children:
        cmpoint = (0, 0)  # maximal child memory

        for idx, (proc, data) in enumerate(chld.items()):
            # Create the numpy arrays from the series data
            cts = np.asarray([item[1] for item in data]) - global_start
            cmem = np.asarray([item[0] for item in data])

            # Plot the line to the figure
            pl.plot(cts, cmem, '+-' + mem_line_colors[(idx + 1) % len(mem_line_colors)], label='child {}'.format(proc))

            # Detect the maximal child memory point
            cmax_mem = cmem.max()
            if cmax_mem > cmpoint[1]:
                cmpoint = (cts[cmem.argmax()], cmax_mem)

        # Add the marker lines for the maximal child memory usage
        pl.vlines(cmpoint[0], pl.ylim()[0] + 0.001, pl.ylim()[1] - 0.001, 'r', '--')
        pl.hlines(cmpoint[1], pl.xlim()[0] + 0.001, pl.xlim()[1] - 0.001, 'r', '--')

    def mouse_motion_handler(event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            for coord, (name, text, rect) in rectangles.items():
                x0, y0, x1, y1 = coord
                if x0 < x < x1 and y0 < y < y1:
                    if FLAME_PLOTTER_VARS['hovered_rect'] == rect:
                        return

                    if FLAME_PLOTTER_VARS['hovered_rect'] is not None:
                        FLAME_PLOTTER_VARS['hovered_rect'].set_alpha(FLAME_PLOTTER_VARS['alpha'])
                        FLAME_PLOTTER_VARS['hovered_text'].set_color((0, 0, 0, 0))
                        FLAME_PLOTTER_VARS['hovered_rect'].set_linewidth(1)

                    FLAME_PLOTTER_VARS['hovered_text'] = text
                    FLAME_PLOTTER_VARS['hovered_rect'] = rect
                    FLAME_PLOTTER_VARS['alpha'] = rect.get_alpha()
                    FLAME_PLOTTER_VARS['hovered_rect'].set_alpha(0.8)
                    FLAME_PLOTTER_VARS['hovered_rect'].set_linewidth(3)
                    FLAME_PLOTTER_VARS['hovered_text'].set_color((0, 0, 0, 1))
                    pl.draw()
                    return

        if FLAME_PLOTTER_VARS['hovered_rect'] is not None:
            FLAME_PLOTTER_VARS['hovered_text'].set_color((0, 0, 0, 0))
            FLAME_PLOTTER_VARS['hovered_rect'].set_alpha(FLAME_PLOTTER_VARS['alpha'])
            FLAME_PLOTTER_VARS['hovered_rect'].set_linewidth(1)
            pl.draw()
            FLAME_PLOTTER_VARS['hovered_rect'] = None
            FLAME_PLOTTER_VARS['hovered_text'] = None

    def mouse_click_handler(event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        for coord, _ in rectangles.items():
            x0, y0, x1, y1 = coord
            if x0 < x < x1 and y0 < y < y1:
                toolbar = pl.gcf().canvas.toolbar
                toolbar.push_current()
                timestamp_ax.set_xlim(x0, x1)
                timestamp_ax.set_ylim(y0, stack_size + 1)
                toolbar.push_current()
                pl.draw()
                return

    # plot timestamps, if any
    if len(ts) > 0 and timestamps:
        func_num = 0
        f_labels = function_labels(ts.keys())
        rectangles = {}
        for f, exec_ts in ts.items():
            for execution in exec_ts:
                x0, x1 = execution[:2]
                y0 = execution[4]
                y1 = y0 + 1
                x0 -= global_start
                x1 -= global_start
                color = next(colors[y0])
                rect, text = add_timestamp_rectangle(timestamp_ax, x0, x1, y0, y1, f, color=color)
                rectangles[(x0, y0, x1, y1)] = (f, text, rect)
            func_num += 1

        # Disable hovering if there are too many rectangle to prevent slow down
        if len(rectangles) < 100:
            pl.gcf().canvas.mpl_connect('motion_notify_event', mouse_motion_handler)
        pl.gcf().canvas.mpl_connect('button_release_event', mouse_click_handler)

    if timestamps:
        pl.hlines(max_mem, pl.xlim()[0] + 0.001, pl.xlim()[1] - 0.001, colors='r', linestyles='--')
        pl.vlines(t[max_mem_ind], bottom, top, colors='r', linestyles='--')

    pl.sca(ax)

    return mprofile


def add_timestamp_rectangle(ax, x0, x1, y0, y1, func_name, color='none'):
    rect = ax.fill_betweenx((y0, y1), x0, x1, color=color, alpha=0.5, linewidth=1)
    text = ax.text(x0, y1, func_name, horizontalalignment='left', verticalalignment='top', color=(0, 0, 0, 0))
    return rect, text


def function_labels(dotted_function_names):
    state = {}

    def set_state_for(function_names, level):
        for fn in function_names:
            label = '.'.join(fn.split('.')[-level:])
            label_state = state.setdefault(label, {'functions': [], 'level': level})
            label_state['functions'].append(fn)

    set_state_for(dotted_function_names, 1)

    while True:
        ambiguous_labels = [label for label in state if len(state[label]['functions']) > 1]
        for ambiguous_label in ambiguous_labels:
            function_names = state[ambiguous_label]['functions']
            new_level = state[ambiguous_label]['level'] + 1
            del state[ambiguous_label]
            set_state_for(function_names, new_level)
        if len(ambiguous_labels) == 0:
            break

    fn_to_label = dict((label_state['functions'][0], label) for label, label_state in state.items())

    return fn_to_label


def filter_mprofile_mem_usage_by_function(prof, func):
    if func is None:
        return prof['mem_usage']

    if func not in prof['func_timestamp']:
        raise ValueError(str(func) + ' was not found.')

    time_ranges = prof['func_timestamp'][func]
    filtered_memory = []

    # The check here could be improved, but it's done in this
    # inefficient way to make sure we don't miss overlapping
    # ranges.
    for mib, ts in zip(prof['mem_usage'], prof['timestamp']):
        for rng in time_ranges:
            if rng[0] <= ts <= rng[1]:
                filtered_memory.append(mib)

    return filtered_memory


def get_profiles(args):
    profiles = sorted(glob.glob('mprofile_????-??-??-??:??:??.dat'))

    if len(args.profiles) == 0:
        if len(profiles) == 0:
            print(color.red(f'No input file found. \nThis program looks for mprofile_*.dat files, generated by the `{color.green('mprof run')}` command.'))
            sys.exit(-1)
        print(color.blue(f'Using latest profile `{profiles[-1]}`'))
        filenames = [profiles[-1]]
    else:
        filenames = []
        for prof in {*args.profiles}:
            if osp.exists(prof):
                filenames.append(prof)
            elif not prof.isdigit():
                raise ValueError(color.red(f'Input file not found: {prof}'))
            elif (n := int(prof)) >= len(profiles):
                raise IndexError(color.red(f"Can't find index {prof} from [0: {len(profiles) - 1}]"))
            else:
                filenames.append(profiles[n])

    if not len(filenames):
        print(color.red('No files found from given input.'))
        sys.exit(-1)

    return filenames


def peak_action():
    desc = """Prints the peak memory used in data file `file.dat` generated
using `mprof run`. If no .dat file is given, it will take the most recent
such file in the current directory."""
    parser = ArgumentParser(usage='mprof peak [options] [file.dat]', description=desc)
    parser.add_argument('profiles', nargs='*', help='profiles made by mprof run')
    parser.add_argument('--func', dest='func', default=None, help="""Show the peak for this function. Does not support child processes.""")
    args = parser.parse_args()
    filenames = get_profiles(args)

    for filename in filenames:
        prof = read_mprofile_file(filename)
        try:
            mem_usage = filter_mprofile_mem_usage_by_function(prof, args.func)
        except ValueError:
            print('{}\tNaN MiB'.format(prof['filename']))
            continue

        print('{}\t{:.3f} MiB'.format(prof['filename'], max(mem_usage)))
        for child, values in prof['children'].items():
            child_peak = max([mem_ts[0] for mem_ts in values])
            print('  Child {}\t\t\t{:.3f} MiB'.format(child, child_peak))


def list_action():
    """Display existing profiles, with indices."""
    parser = ArgumentParser(usage='mprof list\nThis command takes no argument.')
    parser.add_argument('--version', action='version', version=mp.__version__)
    args = parser.parse_args()

    filenames = get_profile_filenames('all')
    for n, filename in enumerate(filenames):
        ts = osp.splitext(filename)[0].split('_')[-1]
        print('{index} {filename} {hour}:{min}:{sec} {day}/{month}/{year}'.format(index=n, filename=filename, year=ts[:4], month=ts[4:6], day=ts[6:8], hour=ts[8:10], min=ts[10:12], sec=ts[12:14]))


def rm_action():
    """TODO: merge with clean_action (@pgervais)"""
    parser = ArgumentParser(usage='mprof rm [options] numbers_or_filenames')
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument('--dry-run', dest='dry_run', default=False, action='store_true', help='Show what will be done, without actually doing it.')
    parser.add_argument('numbers_or_filenames', nargs='*', help='numbers or filenames removed')
    args = parser.parse_args()

    if len(args.numbers_or_filenames) == 0:
        print('A profile to remove must be provided (number or filename)')
        sys.exit(1)

    filenames = get_profile_filenames(args.numbers_or_filenames)
    if args.dry_run:
        print('Files to be removed: ')
        for filename in filenames:
            print(filename)
    else:
        for filename in filenames:
            os.remove(filename)


def clean_action():
    """Remove every profile file in current directory."""
    parser = ArgumentParser(usage='mprof clean\nThis command takes no argument.')
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument('--dry-run', dest='dry_run', default=False, action='store_true', help="""Show what will be done, without actually doing it.""")
    args = parser.parse_args()

    filenames = get_profile_filenames('all')
    if args.dry_run:
        print('Files to be removed: ')
        for filename in filenames:
            print(filename)
    else:
        for filename in filenames:
            os.remove(filename)


def attach_action():
    argv = sys.argv
    sys.argv = argv[:1] + ['--attach'] + argv[1:]
    run_action()


def run_action():
    import subprocess
    parser = ArgumentParser(usage='mprof run [options] program', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument('--python', dest='python', action='store_true', help='Activates extra features when the profiling executable is a Python program (currently: function timestamping.)')
    parser.add_argument('--nopython', dest='nopython', action='store_true', help='Disables extra features when the profiled executable is a Python program (currently: function timestamping.)')
    parser.add_argument('--interval', '-T', dest='interval', default='0.1', type=float, action='store', help='Sampling period (in seconds), defaults to 0.1')
    parser.add_argument('--include-children', '-C', dest='include_children', action='store_true', help='Monitors forked processes as well (sum up all process memory)')
    parser.add_argument('--multiprocess', '-M', dest='multiprocess', action='store_true', help='Monitors forked processes creating individual plots for each child (disables --python features)')
    parser.add_argument('--exit-code', '-E', dest='exit_code', action='store_true', help='Propagate the exit code')
    attach_arg = parser.add_argument('--attach', '-a', dest='attach_existing', action='store_true', help='Attach to an existing process, by process name or by pid')
    parser.add_argument('--timeout', '-t', dest='timeout', action='store', type=int, help='timeout in seconds for the profiling, default new process has no timeout, attach existing is 1 hour')
    parser.add_argument('--output', '-o', dest='filename', default=f'mprofile_{mp.now()}.dat', help='File to store results in, defaults to `mprofile_<YYYY-MM-DD-hh-mm-ss>.dat` in current directory\n'
                                                                                                    '(which is the date-time of the program start).\n'
                                                                                                    'This file contains the process memory consumption, in Mb (one value per line).')
    parser.add_argument('--backend', dest='backend', choices=['psutil', 'psutil_pss', 'psutil_uss', 'posix', 'tracemalloc'], default='psutil',
                        help='Current supported backends: `psutil`, `psutil_pss`, `psutil_uss`, `posix`, `tracemalloc`. Defaults to `psutil`.')
    parser.add_argument('program', nargs=REMAINDER, help='Option 1: "<EXECUTABLE> <ARG1> <ARG2>..." - profile executable\n'
                                                         'Option 2: "<PYTHON_SCRIPT> <ARG1> <ARG2>..." - profile python script\n'
                                                         'Option 3: (--python flag present) "<PYTHON_EXECUTABLE> <PYTHON_SCRIPT> <ARG1> <ARG2>..." - profile python script with specified interpreter\n'
                                                         'Option 4: (--python flag present) "<PYTHON_MODULE> <ARG1> <ARG2>..." - profile python module\n')
    args = parser.parse_args()

    if len(args.program) == 0:
        print('A program to run must be provided. Use -h for help')
        sys.exit(1)

    print(f'{osp.basename(sys.argv[0])}: Sampling memory every {color.blue(f'{args.interval}s')}')

    mprofile_output = args.filename

    program = args.program
    if args.attach_existing:
        print(f'attaching to existing process, using hint: {program[0]}')
        if program[0].isdigit():
            p = literal_eval(program[0])
            cmd_line = get_cmd_line(program)
        else:
            proc = find_first_process(program[0])
            if proc is None:
                raise ArgumentError(attach_arg, f'\nWhen attaching, program should be process name or pid.\nFailed to find a process using hint: {program[0]}')

            p = proc.pid
            try:
                cmd_line = proc.cmdline()
            except:
                cmd_line = get_cmd_line(program)
        if args.timeout is None:
            args.timeout = 3600
    else:
        print('running new process')
        # TODO: more than one script as argument ?
        if program[0].endswith('.py') and not args.nopython:
            if args.multiprocess:
                # in multiprocessing mode you want to spawn a separate python process
                if not program[0].startswith('python'):
                    program.insert(0, sys.executable)
                args.python = False
            else:
                args.python = True
        if args.python:
            print('running as a Python program...')
            if not program[0].startswith('python'):
                program.insert(0, sys.executable)
            cmd_line = get_cmd_line(program)
            extra_args = ['-m', 'memory_profiler', '--timestamp', '-o', mprofile_output]
            if args.include_children:
                extra_args.append('--include-children')
            program[1:1] = extra_args
            p = subprocess.Popen(program)
        else:
            cmd_line = get_cmd_line(program)
            p = subprocess.Popen(program)

    with open(mprofile_output, 'a') as f:
        f.write(f'CMDLINE {cmd_line}\n')
        mp.memory_usage(proc=p, interval=args.interval, timeout=args.timeout, timestamps=True, include_children=args.include_children, multiprocess=args.multiprocess, stream=f, backend=args.backend)

    if args.exit_code:
        if p.returncode != 0:
            logger.error(f'Program resulted with a non-zero exit code: {p.returncode}')
        sys.exit(p.returncode)


def plot_file(filename, timestamps=True, children=True, options=None):
    try:
        import pylab as plt
    except ImportError as e:
        print(color.yellow('matplotlib is needed for plotting.'))
        print(e)
        sys.exit(1)

    mprofile = read_mprofile_file(filename)

    if len(mprofile['timestamp']) == 0:
        print(color.red(f'** No memory usage values have been found in {filename} file.** File may be empty or invalid.\nIt can be deleted with "mprof rm {filename}"'))
        sys.exit(0)

    # Merge function timestamps and memory usage together
    fts = mprofile['func_timestamp']
    ts = mprofile['timestamp']
    mem = mprofile['mem_usage']
    child = mprofile['children']

    if len(fts) > 0:
        for values in fts.values():
            for v in values:
                ts.extend(v[:2])
                mem.extend(v[2:4])

    ts = np.asarray(ts)
    sort_index = ts.argsort()
    ts = ts[sort_index]
    mem = np.asarray(mem)[sort_index]

    # Plot curves
    global_start = float(ts[0])
    ts -= global_start

    max_ts = ts.max()
    max_mem = mem.max()
    show_trend_slope = options is not None and hasattr(options, 'slope') and options.slope is True

    legend = f'{'' if re.search(r'mprofile_*.dat', filename) else filename.split('.', 1)[0]}-[{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(global_start))}'
    mem_line_label = f'{legend}.{int(round(math.modf(global_start)[0] * 1000)):03d}]'

    mem_trend = None
    if show_trend_slope:
        # Compute trend line
        mem_trend = np.polyfit(ts, mem, 2)

    from adjustText import adjust_text

    text = []
    p = plt.plot(ts, mem, ',-', label=mem_line_label)
    main_color, main_linewidth = p[0].get_color(), p[0].get_linewidth()
    text.append(plt.annotate(f'{max_mem:.1f}Mib', xy=(max_ts, max_mem), xytext=(0, 7), textcoords='offset points', color=main_color))

    if show_trend_slope:
        # Plot the trend line
        # print(len(ts), len(mp.polynomial(ts, mem_trend, 1)))
        # exit()
        plt.plot(ts, (slope := mp.polynomial(ts, mem_trend, 2)), '-->', linewidth=main_linewidth / 2, color=main_color, markevery=200)
        text.append(plt.annotate(f'{mem_trend[0]:.1f}x^2 + {mem_trend[1]:.1f}x', xy=(float(ts[len(ts) >> 1]), slope[len(ts) >> 1]), xytext=(0, 7), textcoords='offset points', color=main_color))

    # plot children, if any
    if len(child) > 0 and children:
        cmpoint = (0, 0)  # maximal child memory

        for idx, (proc, data) in enumerate(child.items()):
            # Create the numpy arrays from the series data
            cts = np.asarray([item[1] for item in data]) - global_start
            cmem = np.asarray([item[0] for item in data])
            max_cmem = cmem.max()

            cmem_trend = None
            child_mem_trend_label = filename.split('.', 1)[0]
            if show_trend_slope:
                # Compute trend line
                cmem_trend = np.polyfit(cts, cmem, 2)

            # Plot the line to the figure
            plt.plot(cts, cmem, ',-.', label=f'child {proc}-{child_mem_trend_label}', linewidth=main_linewidth / 2, color=main_color)
            text.append(plt.annotate(f'{max_cmem:.1f}Mib', xy=(cts.max(), max_cmem), xytext=(0, 7), textcoords='offset points', color=main_color))

            if show_trend_slope:
                # Plot the trend line
                plt.plot(cts, (cslope := mp.polynomial(cts, cmem_trend, 2)), '->', linewidth=main_linewidth / 4, color=main_color, markevery=200)
                text.append(plt.annotate(f'{cmem_trend[0]:.1f}x^2+{cmem_trend[1]:.1f}x', xy=(float(cts[len(ts) >> 1]), cslope[len(ts) >> 1]), xytext=(0, 7), textcoords='offset points', color=main_color))

            # Detect the maximal child memory point
            if max_cmem > cmpoint[1]:
                cmpoint = (cts[cmem.argmax()], max_cmem)

        # Add the marker lines for the maximal child memory usage
        plt.vlines(cmpoint[0], plt.ylim()[0] + 0.001, plt.ylim()[1] - 0.001, 'r', '--')
        plt.hlines(cmpoint[1], plt.xlim()[0] + 0.001, plt.xlim()[1] - 0.001, 'r', '--')

    # plot timestamps, if any
    all_colors = ('c', 'y', 'g', 'r', 'b')
    if len(fts) > 0 and timestamps:
        func_num = 0
        f_labels = function_labels(fts.keys())
        for f, exec_ts in fts.items():
            for execution in exec_ts:
                add_brackets(execution[:2], execution[2:], xshift=global_start, color=all_colors[func_num % len(all_colors)], label=f_labels[f] + ' %.3fs' % (execution[1] - execution[0]),
                             options=options)
            func_num += 1

    if timestamps:
        plt.hlines(max_mem, plt.xlim()[0] + 0.001, plt.xlim()[1] - 0.001, colors='r', linestyles='--')
        plt.vlines(ts[mem.argmax()], plt.ylim()[0] + 0.001, plt.ylim()[1] - 0.001, colors='r', linestyles='--')
    adjust_text(text, only_move={'text': 'y+'})
    return mprofile


def plot_action():
    def xlim_type(value):
        try:
            newvalue = [float(x) for x in value.split(',')]
        except:
            raise ArgumentError(None, f'{value} option must contain two numbers separated with a comma')
        if len(newvalue) != 2:
            raise ArgumentError(None, f'{value} option must contain two numbers separated with a comma')
        return newvalue

    desc = 'Plots using matplotlib the data file `file.dat` generated using `mprof run`. If no .dat file is given, it will take the most recent such file in the current directory. '
    parser = ArgumentParser(usage='mprof plot [options] [file.dat]', description=desc)
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument('--title', '-t', dest='title', default=None, type=str, action='store', help='String shown as plot title')
    parser.add_argument('--dpi', '-d', dest='dpi', default=100, type=int, action='store', help='Set plot dpi')
    parser.add_argument('--no-function-ts', '-n', dest='no_timestamps', action='store_true', help='Do not display function timestamps on plot.')
    parser.add_argument('--output', '-o', help='Save plot to file instead of displaying it.')
    parser.add_argument('--window', '-w', dest='xlim', type=xlim_type, help='Plot a time-subset of the data. E.g. to plot between 0 and 20.5 seconds: --window 0,20.5')
    parser.add_argument('--flame', '-f', dest='flame_mode', action='store_true', help='Plot the timestamps as a flame-graph instead of the default brackets')
    parser.add_argument('--slope', '-s', dest='slope', action='store_true', help='Plot a trend line and its numerical slope')
    parser.add_argument('--backend', type=str, help='Specify the Matplotlib backend to use')
    parser.add_argument('profiles', nargs='*', help='profiles made by mprof run')
    args = parser.parse_args()

    try:
        if args.backend:
            import matplotlib
            matplotlib.use(args.backend)

        import matplotlib.pyplot as plt
    except ImportError as e:
        print(color.yellow('matplotlib is needed for plotting.'))
        print(e)
        sys.exit(1)
    plt.ioff()

    filenames = get_profiles(args)

    #  + cycler(linestyle=('-', '--', ':', '-.', '---', ''))
    default_cycler = (cycler(color=('k', 'b', 'r', 'g', 'c', 'y', 'm')))
    fig = plt.figure(figsize=(14, 6), dpi=args.dpi)
    # plt.rcParams['figure.dpi'] =
    plt.rc('axes', prop_cycle=default_cycler)
    if not args.flame_mode:
        ax = fig.add_axes((0.1, 0.1, 0.6, 0.75))
    else:
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    if args.xlim:
        plt.xlim(args.xlim[0], args.xlim[1])

    timestamps = False if len(filenames) > 1 or args.no_timestamps else True
    plotter = flame_plotter if args.flame_mode else plot_file
    cmd_line = ''
    for filename in filenames:
        cmd_line = plotter(filename, timestamps=timestamps, options=args)['cmd_line']

    plt.xlabel('Time / seconds')
    plt.ylabel('Memory / Mib')

    if args.title:
        plt.title(args.title)
    elif len(filenames) == 1:
        plt.title(cmd_line)

    # place legend within the plot, make partially transparent in case it obscures part of the lineplot
    if not args.flame_mode:
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_alpha(0.5)
        plt.grid()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


def get_action():
    """ Pop first argument, check it is a valid action. """
    if len(sys.argv) <= 1 or sys.argv[1] not in ALL_ACTIONS:
        print(f'Usage: {color.green(osp.basename(sys.argv[0]))} {color.white('<command>')} {color.white('<options>')} {color.white('<arguments>')}')
        print(help_msg)
        sys.exit(1)

    return sys.argv.pop(1)


def main():
    # Workaround for optparse limitation: insert -- before first negative number found.
    neg_int = re.compile(r'-[0-9]+')
    for i, arg in enumerate(sys.argv):
        if neg_int.match(arg):
            sys.argv.insert(i, '--')
            break

    actions = {'rm': rm_action, 'clean': clean_action, 'list': list_action, 'run': run_action, 'attach': attach_action, 'plot': plot_action, 'peak': peak_action}
    actions[get_action()]()


ALL_ACTIONS = ('run', 'rm', 'clean', 'list', 'plot', 'attach', 'peak')
help_msg = f'''
Available commands:
    {color.green('run')}      run a given command or python file
    {color.green('attach')}   alias for 'run --attach': attach to an existing process by pid or name
    {color.green('rm')}       remove a given file generated by mprof
    {color.green('clean')}    clean the current directory from files created by mprof
    {color.green('list')}     display existing profiles, with indices
    {color.green('plot')}     plot memory consumption generated by mprof run
    {color.green('peak')}     print the maximum memory used by an mprof run

Type {color.green('mprof')} {color.white('<command>')} {color.blue('--help')} for usage help on a specific command.
For example, mprof plot --help will list all plotting options.
'''

if __name__ == '__main__':
    main()
