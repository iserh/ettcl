from tqdm.autonotebook import tqdm as _tqdm


def tqdm(
    iterable=None,
    desc=None,
    total=None,
    leave=True,
    file=None,
    ncols=None,
    mininterval=0.1,
    maxinterval=10.0,
    miniters=None,
    ascii=None,
    disable=False,
    unit="it",
    unit_scale=False,
    dynamic_ncols=False,
    smoothing=0.3,
    bar_format=None,
    initial=0,
    position=None,
    postfix=None,
    unit_divisor=1000,
    write_bytes=False,
    lock_args=None,
    nrows=None,
    colour=None,
    delay=0,
    gui=False,
    **kwargs,
):
    if not disable:
        return _tqdm(
            iterable=iterable,
            desc=desc,
            total=total,
            leave=leave,
            file=file,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=ascii,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            write_bytes=write_bytes,
            lock_args=lock_args,
            nrows=nrows,
            colour=colour,
            delay=delay,
            gui=gui,
            **kwargs,
        )
    else:
        return iterable


def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)
