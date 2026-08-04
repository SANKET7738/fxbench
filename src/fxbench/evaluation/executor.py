"""Formula execution engine using xlwings."""

import atexit
import hashlib
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Literal, Optional, Union

import pywintypes
import xlwings as xw
from pydantic import BaseModel
from rich.console import Console

# Excel COM rejects file paths whose absolute length approaches Windows
# MAX_PATH (260). When this triggers we transparently copy the workbook into
# a short temp path before opening, then delete the copy on exit.
_MAX_PATH_THRESHOLD = 240
_SHORT_PATH_DIR = Path(tempfile.gettempdir()) / "xlfc_short"

Scalar = Union[str, float, bool, None, datetime, dt_time, timedelta]


class CellValue(BaseModel):
    value: Scalar
    format: Optional[str] = None
    formatted: Optional[str] = None

    def __hash__(self):
        return hash((self.value, self.format, self.formatted))


class ColumnValue(BaseModel):
    cells: list[CellValue]

    def __hash__(self):
        return hash(tuple(self.cells))


class RangeValue(BaseModel):
    cells: list[list[CellValue]]

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.cells))


class ErrorValue(BaseModel):
    error_type: str = "unknown"
    message: str = ""


ExecutedValue = Union[CellValue, ColumnValue, RangeValue, ErrorValue]


class CellLocation(BaseModel):
    kind: Literal["cell"] = "cell"
    address: str
    sheet: Optional[str] = None


# --- Excel application pool ---


def _hide_app(app: xw.App):
    """Make Excel window invisible."""
    import win32con
    import win32gui

    hwnd = app.hwnd
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x80000
    WS_EX_TOOLWINDOW = 0x00000080
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
    )
    ex_style = win32gui.GetWindowLong(hwnd, GWL_EXSTYLE)
    win32gui.SetWindowLong(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED | WS_EX_TOOLWINDOW)
    win32gui.SetLayeredWindowAttributes(hwnd, 0, 0, win32con.LWA_ALPHA)


def _close_app(app: xw.App) -> None:
    """Close a single Excel application instance."""
    import win32con
    import win32gui

    try:
        if win32gui.IsWindow(app.hwnd):
            win32gui.PostMessage(app.hwnd, win32con.WM_CLOSE, 0, 0)
    except Exception:
        pass


def _close_book(book: xw.Book):
    """Close an Excel book safely."""
    if book is None:
        return
    try:
        book.close()
    except (pywintypes.com_error, OSError) as e:
        msg = str(e)
        if "disconnected from its clients" not in msg and "Call was rejected by callee" not in msg:
            raise


class ExcelPool:
    """Singleton thread-safe pool of Excel application instances."""

    size: int = 1
    hide: bool = True
    console: Console = Console(quiet=True)

    _instance: Optional["ExcelPool"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.lock = threading.Lock()
        self.ready: list[xw.App] = []
        self.busy: list[xw.App] = []

    @classmethod
    def setup(cls, size: Optional[int] = None, hide: Optional[bool] = None, console: Optional[Console] = None):
        with cls._lock:
            if size is not None:
                cls.size = size
                if cls._instance is not None and cls._instance.busy:
                    raise ValueError("Cannot change pool size while instances are busy.")
                if cls._instance is not None and len(cls._instance.ready) > size:
                    to_close = cls._instance.ready[size:]
                    cls._instance.ready = cls._instance.ready[:size]
                    for app in to_close:
                        _close_app(app)
            if hide is not None and cls.hide != hide:
                if cls._instance is not None:
                    if cls._instance.busy:
                        raise ValueError("Cannot change visibility while instances are busy.")
                    for app in cls._instance.ready:
                        _close_app(app)
                    cls._instance.ready.clear()
                cls.hide = hide
            if console is not None:
                cls.console = console

    @classmethod
    def instance(cls) -> "ExcelPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        if cls._instance is None:
            return
        with cls._lock:
            for app in cls._instance.busy + cls._instance.ready:
                _close_app(app)
        cls._instance = None

    def acquire(self) -> xw.App:
        while True:
            with self._lock:
                if self.ready:
                    app = self.ready.pop()
                    self.busy.append(app)
                    return app
                if len(self.busy) < self.size:
                    app = xw.App(visible=not self.hide, add_book=False)
                    app.display_alerts = False
                    if self.hide:
                        _hide_app(app)
                    self.busy.append(app)
                    return app
            time.sleep(0.1)

    def release(self, app: xw.App, *, poison: bool = False):
        with self._lock:
            try:
                if app is None:
                    return
                if app in self.busy:
                    self.busy.remove(app)
                if poison:
                    _close_app(app)
                    return
                app.version  # health check
                self.ready.append(app)
            except Exception:
                _close_app(app)


def _short_path_for(file: Path) -> Optional[Path]:
    """Return a short temp-dir path copy when the absolute path exceeds the
    Windows MAX_PATH threshold; otherwise None. The copy is reused across
    calls so repeated openings of the same workbook don't re-copy.
    """
    abs_p = os.path.abspath(file)
    if len(abs_p) < _MAX_PATH_THRESHOLD:
        return None
    _SHORT_PATH_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(abs_p.encode("utf-16-le")).hexdigest()[:16]
    short = _SHORT_PATH_DIR / f"{digest}{file.suffix or '.xlsx'}"
    if not short.exists() or short.stat().st_mtime < Path(abs_p).stat().st_mtime:
        try:
            shutil.copy2(abs_p, short)
        except Exception:
            return None
    return short


class ExcelBook:
    """Context manager for opening and closing an Excel workbook."""

    def __init__(self, file: Path, save: bool = False):
        self.path = file
        self.save = save
        self.app: Optional[xw.App] = None
        self.book: Optional[xw.Book] = None
        self._open_path: Optional[Path] = None  # actually-opened path (may differ from self.path under MAX_PATH copy)
        self._poison_app: bool = False

    def __enter__(self) -> Optional["ExcelBook"]:
        pool = ExcelPool.instance()
        # Attempt sequence: (path, corrupt_load). Each tuple gets a fresh
        # Excel app slot. Escalates from normal open → MAX_PATH copy → repair.
        short = _short_path_for(self.path)
        attempts: list[tuple[Path, int]] = []
        if short is not None:
            attempts.append((short, 0))           # MAX_PATH workaround first
            attempts.append((short, 1))           # then repair-load on the copy
        else:
            attempts.append((self.path, 0))       # normal
            attempts.append((self.path, 0))       # retry with fresh app (COM degradation)
            attempts.append((self.path, 1))       # repair-load as last resort
        last_exc = None
        for open_path, corrupt_load in attempts:
            self.app = pool.acquire()
            try:
                self.book = self.app.books.open(
                    str(open_path),
                    ignore_read_only_recommended=True,
                    update_links=False,
                    notify=False,
                    corrupt_load=corrupt_load,
                )
                self._open_path = open_path
                return self
            except Exception as e:
                last_exc = e
                if self.app is not None:
                    pool.release(self.app, poison=True)
                    self.app = None
        _ = last_exc
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        pool = ExcelPool.instance()
        try:
            if self.book is not None:
                if self.save:
                    try:
                        self.book.save()
                    except pywintypes.com_error:
                        pass
                _close_book(self.book)
        except Exception:
            pass
        finally:
            pool.release(self.app)
            self.book = None
            self.app = None


atexit.register(ExcelPool.clear)


EXCEL_ERRORS: dict[int, str] = {
    -2146826281: "#DIV/0!",
    -2146826246: "#N/A",
    -2146826273: "#VALUE!",
    -2146826259: "#NAME?",
    -2146826265: "#REF!",
    -2146826252: "#NUM!",
    -2146826288: "#NULL!",
}

EXCEL_ERROR_TEXTS: set[str] = set(EXCEL_ERRORS.values())
INFRASTRUCTURE_ERROR_TYPES: set[str] = {
    "parse_error",
    "workbook_error",
    "sheet_error",
    "unknown",
}


def is_excel_error_value(value: ExecutedValue) -> bool:
    return isinstance(value, ErrorValue) and value.error_type in EXCEL_ERROR_TEXTS


def is_infrastructure_error(value: ExecutedValue) -> bool:
    return isinstance(value, ErrorValue) and value.error_type in INFRASTRUCTURE_ERROR_TYPES


def can_execute_value(value: ExecutedValue) -> bool:
    return not isinstance(value, ErrorValue) or not is_infrastructure_error(value)


def execute(formula: str, workbook: Path, worksheet: str, address: str) -> ExecutedValue:
    """Execute a formula in a workbook cell and return the result."""
    formula = f"={formula.lstrip('=')}"
    location = CellLocation(address=address, sheet=worksheet)
    return _execute_cell(formula, workbook, location)


def _execute_cell(formula: str, workbook: Path, location: CellLocation) -> ExecutedValue:
    with ExcelBook(workbook) as book:
        if book is None:
            return ErrorValue(error_type="workbook_error", message="Could not open workbook")
        if location.sheet and location.sheet not in book.book.sheet_names:
            return ErrorValue(error_type="sheet_error", message=f"Sheet '{location.sheet}' not found")
        sheet = book.book.sheets[location.sheet] if location.sheet else book.book.sheets.active
        cell = sheet[location.address]

        # Phase 1: Parse — set the formula in the cell
        try:
            cell.api.Formula2 = formula
        except Exception as e:
            return ErrorValue(error_type="parse_error", message=str(e)[:200])

        # Phase 2: Read — retrieve the computed value
        try:
            raw_value = cell.api.Value

            # Detect Excel error values via display text (e.g. "#DIV/0!", "#N/A").
            # Using cell.api.Text is more reliable than matching integer COM
            # error codes, which could collide with legitimate cell values.
            displayed = str(cell.api.Text)
            if displayed in EXCEL_ERROR_TEXTS:
                return ErrorValue(
                    error_type=displayed,
                    message=f"Excel returned {displayed}",
                )

            if spill := cell.api.SpillingToRange:
                rows = spill.Rows.Count
                cols = spill.Columns.Count
                data = []
                for r in range(1, rows + 1):
                    row_vals = []
                    for c in range(1, cols + 1):
                        sc = spill.Cells(r, c)
                        row_vals.append(CellValue(
                            value=sc.Value,
                            formatted=sc.Text,
                            format=sc.NumberFormat,
                        ))
                    data.append(row_vals)
                return RangeValue(cells=data)
            return CellValue(
                value=raw_value,
                formatted=cell.api.Text,
                format=cell.api.NumberFormat,
            )
        except Exception as e:
            return ErrorValue(error_type="runtime_error", message=str(e)[:200])
