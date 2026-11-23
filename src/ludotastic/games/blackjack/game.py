"""One hand of blackjack with a dealer who stands on soft 17."""

__all__ = ("Blackjack", "BlackjackState", "Move", "hand_total")

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, override

from pydantic import BaseModel, ConfigDict

from ludotastic.engine.game import ActionMode, Game, GameInfo, Outcome, View
from ludotastic.types import Seat

if TYPE_CHECKING:
    from ludotastic.engine.render import Renderer

TARGET: Final = 21
DEALER_STANDS: Final = 17
OPENING_HAND_SIZE: Final = 2
RANK_VALUES: Final = {
    "A": 1,
    **{str(rank): rank for rank in range(2, 11)},
    "J": 10,
    "Q": 10,
    "K": 10,
}
DECK: Final = tuple(rank + suit for suit in "♣♦♥♠" for rank in RANK_VALUES)


def hand_total(cards: Sequence[str]) -> int:
    """Count every ace low, then promote one ace if it fits."""
    ranks = [card[:-1] for card in cards]
    total = sum(RANK_VALUES[rank] for rank in ranks)
    return total + 10 if "A" in ranks and total + 10 <= TARGET else total


class BlackjackState(BaseModel):
    """The player's hand, the dealer's hand and the remaining draw pile."""

    model_config = ConfigDict(frozen=True)

    player: tuple[str, ...]
    dealer: tuple[str, ...]
    deck: tuple[str, ...]
    finished: bool = False
    second_hand: tuple[str, ...] = ()
    active_hand: Literal[0, 1] = 0

    @property
    def hands(self) -> tuple[tuple[str, ...], ...]:
        """Return the hands in playing order."""
        return (self.player, self.second_hand) if self.second_hand else (self.player,)

    def hand_label(self, index: int) -> str:
        """Identify the hand and indicate whose turn it is."""
        if not self.second_hand:
            return "Your hand"
        label = f"Mão {index + 1}"
        if not self.finished and self.active_hand == index:
            label += " — sua vez"
        return label

    @property
    def can_split(self) -> bool:
        """Allow a single split of matching ranks on the opening deal."""
        return (
            not self.finished
            and not self.second_hand
            and len(self.player) == OPENING_HAND_SIZE
            and self.player[0][:-1] == self.player[1][:-1]
        )


class Move(BaseModel):
    """The player's choice."""

    model_config = ConfigDict(frozen=True)

    action: Literal["hit", "stand", "split"]


class Blackjack(Game[BlackjackState, Move]):
    """Play one hand against the house."""

    info = GameInfo(
        id="blackjack",
        name="Blackjack",
        description="Beat the dealer without going over 21.",
        action_modes=(ActionMode.CHOICE, ActionMode.STRUCTURED),
        views=(View.TEXT,),
        tags=("classic", "cards", "single-player"),
    )
    rules = (
        "Play one hand with a standard 52-card deck. Hit to take a card or stand "
        "to let the dealer play. Face cards count as 10 and aces as 1 or 11. "
        "An opening 21 is blackjack and wins unless both hands have blackjack. "
        "Reaching 21 later automatically stands. Going over 21 loses immediately. "
        "The dealer draws to 17 and stands on soft 17. Equal totals push. "
        "Split an opening pair of identical ranks once to play two hands in order. "
        "Each gets one card immediately. Split aces may hit normally. "
        "After a split, 21 is an ordinary 21. Both hands face the same dealer."
    )
    state_type = BlackjackState
    action_type = Move
    hide_live_seed = True

    @override
    def initial_state(self, rng: random.Random) -> BlackjackState:
        """Deal alternately and settle opening naturals immediately."""
        deck = list(DECK)
        rng.shuffle(deck)
        player = (deck[0], deck[2])
        dealer = (deck[1], deck[3])
        return BlackjackState(
            player=player,
            dealer=dealer,
            deck=tuple(deck[4:]),
            finished=TARGET in (hand_total(player), hand_total(dealer)),
        )

    @override
    def to_move(self, state: BlackjackState) -> Sequence[Seat]:
        return () if state.finished else (0,)

    @override
    def legal_actions(self, state: BlackjackState, player: Seat) -> Sequence[Move]:
        if player not in self.to_move(state):
            return ()
        moves = [Move(action="hit"), Move(action="stand")]
        if state.can_split:
            moves.append(Move(action="split"))
        return moves

    @override
    def apply(
        self, state: BlackjackState, actions: Mapping[int, Move], rng: random.Random
    ) -> BlackjackState:
        """Resolve a player action and any forced dealer draws."""
        del rng
        move = actions[0].action
        hands = list(state.hands)
        active = state.active_hand
        deck = state.deck
        if move == "split":
            if not state.can_split:
                msg = "Only an opening pair of the same rank can split."
                raise ValueError(msg)
            hands = [
                (card, draw) for card, draw in zip(state.player, deck[:2], strict=True)
            ]
            deck = deck[2:]
        elif move == "hit":
            hands[active] += (deck[0],)
            deck = deck[1:]
        else:
            active += 1

        while active < len(hands) and hand_total(hands[active]) >= TARGET:
            active += 1
        finished = active == len(hands)
        dealer = state.dealer
        if finished and any(hand_total(hand) <= TARGET for hand in hands):
            while hand_total(dealer) < DEALER_STANDS:
                dealer += (deck[0],)
                deck = deck[1:]
        return BlackjackState(
            player=hands[0],
            second_hand=hands[1] if len(hands) > 1 else (),
            active_hand=1 if active == 1 and not finished else 0,
            dealer=dealer,
            deck=deck,
            finished=finished,
        )

    @staticmethod
    def _result(player: int, dealer: int) -> tuple[float, str]:
        if player > TARGET:
            score, reason = 0.0, "You bust"
        elif dealer > TARGET:
            score, reason = 1.0, "Dealer busts — you win"
        elif player == dealer:
            score, reason = 0.5, "Push"
        elif player > dealer:
            score, reason = 1.0, "You win"
        else:
            score, reason = 0.0, "Dealer wins"
        return score, reason

    @override
    def outcome(self, state: BlackjackState) -> Outcome | None:
        if not state.finished:
            return None
        player, dealer = hand_total(state.player), hand_total(state.dealer)
        score, reason = self._result(player, dealer)
        if state.second_hand:
            second = hand_total(state.second_hand)
            other, _ = self._result(second, dealer)
            labels = {1.0: "vitória", 0.5: "empate", 0.0: "derrota"}
            return Outcome(
                scores=((score + other) / 2,),
                reason=(
                    f"Mão 1: {labels[score]} ({player}) · "
                    f"Mão 2: {labels[other]} ({second})"
                    f" · Banca: {dealer}"
                ),
            )
        natural = any(
            len(hand) == OPENING_HAND_SIZE and hand_total(hand) == TARGET
            for hand in (state.player, state.dealer)
        )
        prefix = "Blackjack! " if natural else ""
        return Outcome(
            scores=(score,), reason=f"{prefix}{reason} (you {player}, dealer {dealer})."
        )

    @override
    def out_of_turns(self, state: BlackjackState) -> Outcome:
        return Outcome(
            scores=(0.0,), reason="Turn limit reached before the hand ended."
        )

    @override
    def public_state(self, state: BlackjackState) -> BlackjackState:
        dealer = state.dealer if state.finished else state.dealer[:1]
        return state.model_copy(update={"dealer": dealer, "deck": ()})

    @override
    def observe(self, state: BlackjackState, player: Seat) -> str:
        visible = self.public_state(state)
        dealer = " ".join(visible.dealer)
        dealer += (
            f" ({hand_total(visible.dealer)})" if visible.finished else " [face down]"
        )
        lines = [
            f"{visible.hand_label(i)}: {' '.join(hand)} ({hand_total(hand)})"
            for i, hand in enumerate(visible.hands)
        ]
        return "\n".join([*lines, f"Dealer: {dealer}"])

    @override
    def describe_action(self, action: Move) -> str:
        return "Dividir" if action.action == "split" else action.action.title()

    @override
    def renderer(self) -> Renderer:
        """Show the cards on the blackjack table."""
        from ludotastic.games.blackjack.ui import render  # noqa: PLC0415

        return render
