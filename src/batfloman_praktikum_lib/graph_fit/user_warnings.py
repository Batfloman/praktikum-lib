import os
import traceback

from batfloman_praktikum_lib.io.termColors import bcolors

def _find_trace_frame_outside_lib():
    import batfloman_praktikum_lib as bpl

    lib_file = os.path.abspath(bpl.__file__)
    lib_dir = os.path.dirname(lib_file)

    stack = traceback.extract_stack()

    user_frame = None
    for frame in reversed(stack):
        file_dir = os.path.dirname(os.path.abspath(frame.filename))
        if lib_dir not in file_dir:
            return frame

    return stack[-1]

def warn_user_no_y_errors_least_squares(y_data, y_err, ignore_y_errors: bool
):
    if ignore_y_errors or (y_err is not None):
        return

    has_errors = False
    try:
        # z.B. iterierbare Messwerte mit .error
        has_errors = all(hasattr(y, "error") and y.error is not None for y in y_data)
    except TypeError:
        # z.B. Skalar mit .error
        has_errors = hasattr(y_data, "error") and y_data.error is not None

    if not has_errors:
        frame = _find_trace_frame_outside_lib()

        print(f"{bcolors.WARNING}Warning: no y-value uncertainties were detected, using equal weights !{bcolors.ENDC}")
        # print(f"{bcolors.OKBLUE}{bcolors.BOLD} At Line {frame.lineno}{bcolors.ENDC}: `{frame.line}`")
        # print(f"\tin {frame.filename}:{frame.lineno}:0")
        print(f"{bcolors.OKBLUE}{bcolors.BOLD} At Line {frame.lineno}{bcolors.ENDC}: `{frame.filename}:{frame.lineno}:0`")
        print(f"\t{frame.line}")

        print(f"{bcolors.WARNING} - Call with `ignore_warning_y_error = True` to surpress this warning!{bcolors.ENDC}")

def warn_user_x_errors_least_squares(x_data, ignore_x_errors: bool):
    if ignore_x_errors:
        return

    from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
    has_x_err = any(isinstance(x, MeasurementBase) and x.error is not None for x in x_data)
    if has_x_err and not ignore_x_errors:
        frame = _find_trace_frame_outside_lib()

        print(f"{bcolors.WARNING}Warning: x-value uncertainties were detected but are ignored by least-squares fitting!{bcolors.ENDC}")
        # print(f"{bcolors.OKBLUE}{bcolors.BOLD} At Line {frame.lineno}{bcolors.ENDC}: `{frame.line}`")
        # print(f"\tin {frame.filename}:{frame.lineno}:0")
        print(f"{bcolors.OKBLUE}{bcolors.BOLD} At Line {frame.lineno}{bcolors.ENDC}: `{frame.filename}:{frame.lineno}:0`")
        print(f"\t{frame.line}")

        print(f"{bcolors.WARNING} - Use ODR-Fit if x-errors should be included!{bcolors.ENDC}")
        print(f"{bcolors.WARNING} - Call with `ignore_warning_x_error = True` to surpress this warning!{bcolors.ENDC}")
