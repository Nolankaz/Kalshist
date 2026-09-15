"""Exact Kalshi event-contract fee calculations for the KXBTC15M model.

The current primary project route is a directly clearing Predictions member. The
historical use of these current mechanics remains assumption A5 in
``execution_notes.md`` and is not proven for every date in the evaluation window.
"""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, localcontext
from numbers import Integral


TAKER_FEE_COEFFICIENT = Decimal("0.07")
TRADE_FEE_QUANTUM = Decimal("0.000001")
DIRECT_MEMBER_BALANCE_QUANTUM = Decimal("0.0001")
NON_DIRECT_FCM_BALANCE_QUANTUM = Decimal("0.01")

KXBTC15M_FEE_TYPE = "quadratic"
KXBTC15M_FEE_MULTIPLIER = Decimal("1")

DIRECT_MEMBER = "direct_member"
NON_DIRECT_FCM = "non_direct_fcm"
SUPPORTED_FEE_TYPES = frozenset({KXBTC15M_FEE_TYPE})
SUPPORTED_ACTIONS = frozenset({"buy", "sell"})
SUPPORTED_ROUNDING_MODES = frozenset({DIRECT_MEMBER, NON_DIRECT_FCM})

FEE_SCHEDULE_SOURCE_URL = "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
FEE_ROUNDING_SOURCE_URL = "https://docs.kalshi.com/getting_started/fee_rounding.md"
FEE_SOURCE_RETRIEVAL_DATE = "2026-09-14"

LOWER_TAPER_BOUND = Decimal("0.10")
UPPER_TAPER_BOUND = Decimal("0.90")
DECI_CENT_TICK = Decimal("0.001")
CENT_TICK = Decimal("0.01")
ZERO = Decimal("0")
ONE = Decimal("1")


def _to_decimal(value, name):
    """Convert a numeric input without importing binary-float error."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite decimal-compatible number")

    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, float):
            result = Decimal(str(value))
        elif isinstance(value, Integral):
            result = Decimal(int(value))
        elif isinstance(value, str):
            result = Decimal(value)
        else:
            raise TypeError(f"{name} must be a Decimal, int, float, or numeric string")
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a valid decimal number") from exc

    if not result.is_finite():
        raise ValueError(f"{name} must be finite")

    return result


def _decimal_width(value):
    """Return a conservative significant-digit width for exact local arithmetic."""
    _, digits, exponent = value.as_tuple()
    if exponent >= 0:
        return len(digits) + exponent
    return max(len(digits), -exponent + 1)


def _validate_price(price):
    price_decimal = _to_decimal(price, "price")
    if not ZERO < price_decimal < ONE:
        raise ValueError("price must satisfy 0 < price < 1")
    return price_decimal


def _validate_contracts(contracts):
    if isinstance(contracts, bool) or not isinstance(contracts, Integral):
        raise TypeError("contracts must be a positive integer")
    contracts_integer = int(contracts)
    if contracts_integer <= 0:
        raise ValueError("contracts must be a positive integer")
    return contracts_integer


def _validate_multiplier(multiplier):
    multiplier_decimal = _to_decimal(multiplier, "multiplier")
    if multiplier_decimal < ZERO:
        raise ValueError("multiplier must be nonnegative")
    return multiplier_decimal


def _validate_fee_type(fee_type):
    if not isinstance(fee_type, str) or fee_type not in SUPPORTED_FEE_TYPES:
        raise ValueError(f"Unsupported fee type: {fee_type!r}")
    return fee_type


def _validate_action(action):
    if not isinstance(action, str) or action not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported action: {action!r}")
    return action


def _balance_quantum(rounding_mode):
    if rounding_mode == DIRECT_MEMBER:
        return DIRECT_MEMBER_BALANCE_QUANTUM
    if rounding_mode == NON_DIRECT_FCM:
        return NON_DIRECT_FCM_BALANCE_QUANTUM
    raise ValueError(f"Unsupported rounding mode: {rounding_mode!r}")


def _floor_to_quantum(value, quantum):
    precision = max(50, _decimal_width(value) + _decimal_width(quantum) + 10)
    with localcontext() as context:
        context.prec = precision
        units = (value / quantum).to_integral_value(rounding=ROUND_FLOOR)
        return units * quantum


def is_on_tick_grid(price):
    """Return whether a price is executable on the KXBTC15M tapered tick grid."""
    price_decimal = _to_decimal(price, "price")
    if not ZERO < price_decimal < ONE:
        return False
    tick = CENT_TICK if LOWER_TAPER_BOUND <= price_decimal <= UPPER_TAPER_BOUND else DECI_CENT_TICK
    return price_decimal % tick == ZERO


def quadratic_model_fee(price, contracts, multiplier=KXBTC15M_FEE_MULTIPLIER, fee_type=KXBTC15M_FEE_TYPE):
    """Return the unrounded quadratic model fee in dollars."""
    price_decimal = _validate_price(price)
    contracts_integer = _validate_contracts(contracts)
    multiplier_decimal = _validate_multiplier(multiplier)
    _validate_fee_type(fee_type)

    precision = max(50, _decimal_width(price_decimal) * 2 + _decimal_width(multiplier_decimal) + len(str(contracts_integer)) + 20)
    with localcontext() as context:
        context.prec = precision
        return multiplier_decimal * TAKER_FEE_COEFFICIENT * contracts_integer * price_decimal * (ONE - price_decimal)


def round_trade_fee(model_fee):
    """Round a nonnegative raw model fee upward to the documented microdollar."""
    model_fee_decimal = _to_decimal(model_fee, "model_fee")
    if model_fee_decimal < ZERO:
        raise ValueError("model_fee must be nonnegative")

    precision = max(50, _decimal_width(model_fee_decimal) + _decimal_width(TRADE_FEE_QUANTUM) + 10)
    with localcontext() as context:
        context.prec = precision
        units = (model_fee_decimal / TRADE_FEE_QUANTUM).to_integral_value(rounding=ROUND_CEILING)
        return units * TRADE_FEE_QUANTUM


@dataclass
class FeeAccumulator:
    """Mutable rounding-overpayment state belonging to exactly one order."""

    unrebated_rounding: Decimal = field(default_factory=lambda: ZERO)
    rounding_mode: str | None = None

    def __post_init__(self):
        self.unrebated_rounding = _to_decimal(self.unrebated_rounding, "unrebated_rounding")
        if self.unrebated_rounding < ZERO:
            raise ValueError("unrebated_rounding must be nonnegative")
        if self.rounding_mode is not None:
            _balance_quantum(self.rounding_mode)


@dataclass(frozen=True)
class FillFeeResult:
    """Auditable components of one fill's principal, rounding, and cash fee."""

    executable_price: Decimal
    contracts: int
    action: str
    fee_type: str
    multiplier: Decimal
    rounding_mode: str
    balance_quantum: Decimal
    model_fee: Decimal
    trade_fee: Decimal
    signed_principal_cash_flow: Decimal
    pre_rebate_posted_cash_change: Decimal
    rounding_adjustment: Decimal
    rebate: Decimal
    net_cash_fee: Decimal
    posted_cash_change: Decimal
    accumulator_remaining: Decimal


def calculate_fill_fee(price, contracts, action, accumulator, multiplier=KXBTC15M_FEE_MULTIPLIER,
                       fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER, validate_tick_grid=True):
    """Calculate one fill and update its order-local accumulator in place."""
    price_decimal = _validate_price(price)
    contracts_integer = _validate_contracts(contracts)
    action = _validate_action(action)
    multiplier_decimal = _validate_multiplier(multiplier)
    fee_type = _validate_fee_type(fee_type)
    balance_quantum = _balance_quantum(rounding_mode)

    if not isinstance(validate_tick_grid, bool):
        raise TypeError("validate_tick_grid must be a bool")

    if not isinstance(accumulator, FeeAccumulator):
        raise TypeError("accumulator must be a FeeAccumulator")
    accumulator.unrebated_rounding = _to_decimal(accumulator.unrebated_rounding, "unrebated_rounding")
    if accumulator.unrebated_rounding < ZERO:
        raise ValueError("unrebated_rounding must be nonnegative")
    if accumulator.rounding_mode is not None and accumulator.rounding_mode != rounding_mode:
        raise ValueError("rounding mode cannot change within one order accumulator")
    if validate_tick_grid and not is_on_tick_grid(price_decimal):
        raise ValueError(f"price {price_decimal} is off the KXBTC15M executable tick grid")

    model_fee = quadratic_model_fee(price_decimal, contracts_integer, multiplier_decimal, fee_type)
    trade_fee = round_trade_fee(model_fee)
    precision = max(50, _decimal_width(price_decimal) + _decimal_width(trade_fee) +
                    _decimal_width(accumulator.unrebated_rounding) + len(str(contracts_integer)) + 20)

    with localcontext() as context:
        context.prec = precision
        unsigned_principal = price_decimal * contracts_integer
        signed_principal = -unsigned_principal if action == "buy" else unsigned_principal
        balance_before_rebate = signed_principal - trade_fee
        pre_rebate_cash_change = _floor_to_quantum(balance_before_rebate, balance_quantum)
        rounding_adjustment = balance_before_rebate - pre_rebate_cash_change
        accumulated_rounding = accumulator.unrebated_rounding + rounding_adjustment
        available_rebate = _floor_to_quantum(accumulated_rounding, balance_quantum)
        gross_fill_fee = trade_fee + rounding_adjustment
        nonnegative_fee_cap = _floor_to_quantum(gross_fill_fee, balance_quantum)
        rebate = min(available_rebate, nonnegative_fee_cap)
        accumulator_remaining = accumulated_rounding - rebate
        net_cash_fee = gross_fill_fee - rebate
        posted_cash_change = pre_rebate_cash_change + rebate
        expected_cash_change = signed_principal - net_cash_fee

    if net_cash_fee < ZERO or posted_cash_change != expected_cash_change:
        raise ArithmeticError("fee calculation invariant failed")

    accumulator.unrebated_rounding = accumulator_remaining
    if accumulator.rounding_mode is None:
        accumulator.rounding_mode = rounding_mode

    return FillFeeResult(
        executable_price=price_decimal,
        contracts=contracts_integer,
        action=action,
        fee_type=fee_type,
        multiplier=multiplier_decimal,
        rounding_mode=rounding_mode,
        balance_quantum=balance_quantum,
        model_fee=model_fee,
        trade_fee=trade_fee,
        signed_principal_cash_flow=signed_principal,
        pre_rebate_posted_cash_change=pre_rebate_cash_change,
        rounding_adjustment=rounding_adjustment,
        rebate=rebate,
        net_cash_fee=net_cash_fee,
        posted_cash_change=posted_cash_change,
        accumulator_remaining=accumulator_remaining,
    )


def calculate_single_fill_fee(price, contracts, action, multiplier=KXBTC15M_FEE_MULTIPLIER,
                              fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER, validate_tick_grid=True):
    """Calculate the project's fresh-order, one-fill historical abstraction."""
    accumulator = FeeAccumulator(rounding_mode=rounding_mode)
    return calculate_fill_fee(price, contracts, action, accumulator, multiplier, fee_type, rounding_mode, validate_tick_grid)


__all__ = [
    "CENT_TICK",
    "DECI_CENT_TICK",
    "DIRECT_MEMBER",
    "DIRECT_MEMBER_BALANCE_QUANTUM",
    "FEE_ROUNDING_SOURCE_URL",
    "FEE_SCHEDULE_SOURCE_URL",
    "FEE_SOURCE_RETRIEVAL_DATE",
    "FeeAccumulator",
    "FillFeeResult",
    "KXBTC15M_FEE_MULTIPLIER",
    "KXBTC15M_FEE_TYPE",
    "NON_DIRECT_FCM",
    "NON_DIRECT_FCM_BALANCE_QUANTUM",
    "SUPPORTED_ACTIONS",
    "SUPPORTED_FEE_TYPES",
    "SUPPORTED_ROUNDING_MODES",
    "TAKER_FEE_COEFFICIENT",
    "TRADE_FEE_QUANTUM",
    "calculate_fill_fee",
    "calculate_single_fill_fee",
    "is_on_tick_grid",
    "quadratic_model_fee",
    "round_trade_fee",
]
