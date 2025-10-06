import logging
import sys

def logging_to_file(log_file):
    logger = logging.getLogger()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )

    # file_log
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    # formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # stream_log
    handler = logging.StreamHandler(sys.stdout)  # 往屏幕上输出
    handler.setLevel(logging.INFO)
    # formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log level
    logger.setLevel(logging.DEBUG)

def collect_monitor(monitor, monitor_a):
    if monitor is None:
        monitor = dict()
        for name, value in monitor_a.items():
            monitor[name] = value
        return  monitor
    else:
        for name, value in monitor.items():
            monitor[name] = value + monitor_a[name]
        return monitor

def average_monitor(monitor, step):
    monitor_new = {}
    for name, value in monitor.items():
        monitor_new[name] = monitor[name] / step
    return  monitor_new

def string_monitor(monitor):
    ss = []
    for name in sorted(monitor):
        value = monitor[name]
        name_num = name + "_num"
        if name_num in monitor:
            ss.append("%s:%0.4f" % (name, float(value / monitor[name_num])))
        elif name.endswith("_num"):
            continue
        elif name.startswith("t_"):
            continue
        else:
            ss.append("%s:%0.4f"%(name, float(value)))
    return  ", ".join(ss)