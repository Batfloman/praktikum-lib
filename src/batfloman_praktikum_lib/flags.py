import argparse
from dataclasses import dataclass

parser = argparse.ArgumentParser()

parser.add_argument(
    "-q", "--quiet",
    action="store_true",
    help="Keine Ausgaben"
)
parser.add_argument(
    "-s", "--skip-popups",
    type=int,
    default=0,
    metavar="N",
    help="Überspringt die ersten N Popup-Sequenzen"
)


@dataclass
class _PopupRuntimeState:
    quiet: bool
    skip_popups_remaining: int


_runtime_state: _PopupRuntimeState | None = None


def _get_runtime_state() -> _PopupRuntimeState:
    global _runtime_state
    if _runtime_state is None:
        args, _ = parser.parse_known_args()
        _runtime_state = _PopupRuntimeState(
            quiet=args.quiet,
            skip_popups_remaining=max(args.skip_popups, 0),
        )
    return _runtime_state

def check_quiet():
    return _get_runtime_state().quiet


def should_skip_popup_sequence() -> bool:
    state = _get_runtime_state()
    if state.quiet:
        return True
    if state.skip_popups_remaining <= 0:
        return False
    state.skip_popups_remaining -= 1
    return True


def _reset_runtime_state_for_tests() -> None:
    global _runtime_state
    _runtime_state = None
