from dataclasses import dataclass
from itertools import permutations
from typing import Optional
import numpy as np


PLAYERS = (0, 1)
CARDS = ("J", "Q", "K")
CARD_RANK = {"J": 0, "Q": 1, "K": 2}

OMEGA = tuple(permutations(CARDS, 2))
CHANCE_PROB = {omega: 1 / len(OMEGA) for omega in OMEGA}

CHECK_CALL = "c"
BET = "b"
FOLD = "f"

ACTIONS = (CHECK_CALL, BET, FOLD)
NON_TERMINAL_HISTORIES = {"", "c", "b", "cb"}
TERMINAL_HISTORIES = {"cc", "bc", "bf", "cbc", "cbf"}
HISTORIES = NON_TERMINAL_HISTORIES | TERMINAL_HISTORIES


def is_terminal(history: str) -> bool:
    """Test whether h is a terminal history z in Z."""
    return history in TERMINAL_HISTORIES


def legal_actions(history: str) -> list[str]:
    """Return A(h), the legal action set after history h."""
    if history in ("", "c"):
        return ["c", "b"]
    if history in ("b", "cb"):
        return ["c", "f"]
    if is_terminal(history):
        return []
    raise ValueError(f"Unknown history: {history!r}")


def next_history(history: str, action: str) -> str:
    """Return h·a for a legal action a in A(h)."""
    if action not in legal_actions(history):
        raise ValueError(f"Illegal action {action!r} after history {history!r}")
    return history + action


def current_player(history: str) -> Optional[int]:
    """Return P(h), the player acting at non-terminal history h."""
    if is_terminal(history):
        return None
    if history in ("", "cb"):
        return 0
    if history in ("c", "b"):
        return 1
    raise ValueError(f"Unknown history: {history!r}")


def showdown_sign_player0(cards: tuple[str, str]) -> int:
    """Return sign(rank(c0)-rank(c1)) for player 0 at showdown."""
    return 1 if CARD_RANK[cards[0]] > CARD_RANK[cards[1]] else -1


def payoff_player0(cards: tuple[str, str], history: str) -> int:
    """Compute u_0(z) for terminal history z and card deal ω."""
    if not is_terminal(history):
        raise ValueError(f"Non-terminal history: {history!r}")
    if history == "cc":
        return showdown_sign_player0(cards)
    if history in ("bc", "cbc"):
        return 2 * showdown_sign_player0(cards)
    if history == "bf":
        return 1
    return -1


def payoff(
    cards: tuple[str, str],
    history: str,
    player: int = 0,
) -> int:
    """Compute zero-sum utility u_i(z), with u_1(z) = -u_0(z)."""
    if player not in PLAYERS:
        raise ValueError(f"Unknown player: {player}")
    utility_0 = payoff_player0(cards, history)
    return utility_0 if player == 0 else -utility_0


def infoset_key(
    cards: tuple[str, str],
    history: str,
) -> Optional[str]:
    """Encode information set I = (player, private card, public history)."""
    player = current_player(history)
    if player is None:
        return None
    return f"P{player}|{cards[player]}|{history}"


@dataclass(frozen=True)
class State:
    """Node h with fixed chance outcome ω = (card_0, card_1)."""

    cards: tuple[str, str]
    history: str = ""

    @property
    def player(self) -> Optional[int]:
        """Return P(h)."""
        return current_player(self.history)

    @property
    def terminal(self) -> bool:
        """Return whether h belongs to Z."""
        return is_terminal(self.history)

    def actions(self) -> list[str]:
        """Return A(h)."""
        return legal_actions(self.history)

    def child(self, action: str) -> "State":
        """Return the successor node h·a."""
        return State(self.cards, next_history(self.history, action))

    def infoset_key(self) -> Optional[str]:
        """Return the information set key I(h)."""
        return infoset_key(self.cards, self.history)

    def utility(self, player: int = 0) -> int:
        """Return u_i(z) for terminal state z."""
        return payoff(self.cards, self.history, player)


Strategy = dict[str, dict[str, float]]


def run_game(
    strategy_0: Strategy,
    strategy_1: Strategy,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> int:
    """Sample one trajectory z ~ σ and return u_0(z)."""
    rng = np.random.default_rng(seed)
    cards = OMEGA[rng.integers(len(OMEGA))]
    state = State(cards)
    strategies = (strategy_0, strategy_1)

    if verbose:
        print(f"Cards: P0={cards[0]}, P1={cards[1]}")

    while not state.terminal:
        player = state.player
        key = state.infoset_key()
        actions = state.actions()
        action_probs = strategies[player][key]
        probabilities = np.array(
            [action_probs[action] for action in actions],
            dtype=float,
        )

        if np.any(probabilities < 0) or not np.isclose(
            probabilities.sum(), 1.0
        ):
            raise ValueError(
                f"Invalid strategy at {key}: {action_probs}"
            )

        action = rng.choice(actions, p=probabilities)

        if verbose:
            print(
                f"{key}: {action_probs} -> {action}"
            )

        state = state.child(action)

    if verbose:
        print(
            f"History: {state.history}, "
            f"utility P0: {state.utility(0)}"
        )

    return state.utility(0)
