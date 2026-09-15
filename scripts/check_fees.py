"""Standalone checks for the exact KXBTC15M fee helpers."""

from dataclasses import fields
from decimal import Decimal
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fees as fees
from scripts.fees import (
    DIRECT_MEMBER,
    DIRECT_MEMBER_BALANCE_QUANTUM,
    KXBTC15M_FEE_TYPE,
    NON_DIRECT_FCM,
    NON_DIRECT_FCM_BALANCE_QUANTUM,
    TRADE_FEE_QUANTUM,
    FeeAccumulator,
    FillFeeResult,
    calculate_fill_fee,
    calculate_single_fill_fee,
    is_on_tick_grid,
    quadratic_model_fee,
    round_trade_fee,
)


OFFICIAL_PDF_PRINTED_REFERENCE = (
    (Decimal("0.50"), 1, Decimal("0.02")),
    (Decimal("0.50"), 100, Decimal("1.75")),
    (Decimal("0.40"), 100, Decimal("1.68")),
    (Decimal("0.10"), 100, Decimal("0.63")),
)

CONTRACT_COUNTS = (1, 2, 4, 5, 10, 25, 100)


def assert_raises(expected_exception, call):
    try:
        call()
    except expected_exception:
        return
    except Exception as exc:
        raise AssertionError(f"Expected {expected_exception.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"Expected {expected_exception.__name__}")


def all_executable_prices():
    lower_tail = [Decimal(mils) / 1000 for mils in range(1, 100)]
    core = [Decimal(cents) / 100 for cents in range(10, 91)]
    upper_tail = [Decimal(mils) / 1000 for mils in range(901, 1000)]
    return lower_tail + core + upper_tail


def print_official_pdf_reference():
    print("Official PDF printed reference (not the Direct Member cash-fee oracle):")
    for price, contracts, printed_fee in OFFICIAL_PDF_PRINTED_REFERENCE:
        print(f"  P={price}, C={contracts}: printed fee=${printed_fee}")
    print("  Checks below validate the raw model and current account-specific cash mechanics separately.\n")


def check_public_api():
    names = (
        "quadratic_model_fee",
        "round_trade_fee",
        "is_on_tick_grid",
        "calculate_fill_fee",
        "calculate_single_fill_fee",
        "FeeAccumulator",
        "FillFeeResult",
    )
    for name in names:
        assert hasattr(fees, name)
    for name in names[:5]:
        assert callable(getattr(fees, name))
    assert set(names).issubset(set(fees.__all__))
    print("PASS: public fee API")


def check_raw_model_fee():
    expected = (
        (Decimal("0.50"), 1, Decimal("1"), Decimal("0.0175")),
        (Decimal("0.50"), 100, Decimal("1"), Decimal("1.75")),
        (Decimal("0.40"), 100, Decimal("1"), Decimal("1.68")),
        (Decimal("0.10"), 100, Decimal("1"), Decimal("0.63")),
        (Decimal("0.005"), 1, Decimal("1"), Decimal("0.00034825")),
        (Decimal("0.995"), 1, Decimal("1"), Decimal("0.00034825")),
        (Decimal("0.50"), 1, Decimal("0"), Decimal("0")),
        (Decimal("0.50"), 4, Decimal("1.5"), Decimal("0.105")),
    )
    for price, contracts, multiplier, expected_fee in expected:
        actual = quadratic_model_fee(price, contracts, multiplier=multiplier)
        assert isinstance(actual, Decimal)
        assert actual == expected_fee

    prices = all_executable_prices()
    for price in prices:
        complement = Decimal("1") - price
        for contracts in CONTRACT_COUNTS:
            assert quadratic_model_fee(price, contracts) == quadratic_model_fee(complement, contracts)

    print(f"PASS: raw quadratic model and symmetry ({len(prices) * len(CONTRACT_COUNTS):,} pairs)")


def check_float_trap_protection():
    cases = (
        (0.50, 4, Decimal("0.07")),
        (0.50, 100, Decimal("1.75")),
        (0.40, 100, Decimal("1.68")),
        (0.10, 100, Decimal("0.63")),
        (0.20, 25, Decimal("0.28")),
        (0.60, 25, Decimal("0.42")),
    )
    for float_price, contracts, expected_raw in cases:
        raw_from_float = quadratic_model_fee(float_price, contracts)
        raw_from_string = quadratic_model_fee(str(float_price), contracts)
        assert raw_from_float == raw_from_string == expected_raw
        assert round_trade_fee(raw_from_float) == round_trade_fee(expected_raw)

    print("PASS: binary-float trap protection")


def check_trade_fee_rounding():
    cases = (
        (Decimal("0"), Decimal("0")),
        (Decimal("0.003638"), Decimal("0.003638")),
        (Decimal("0.0036380000"), Decimal("0.003638")),
        (Decimal("0.0036380001"), Decimal("0.003639")),
        (Decimal("0.0036379999"), Decimal("0.003638")),
        (Decimal("0.00363825"), Decimal("0.003639")),
    )
    for raw, expected in cases:
        rounded = round_trade_fee(raw)
        assert rounded == expected
        assert rounded >= raw
        if raw > 0:
            assert rounded - raw < TRADE_FEE_QUANTUM

    assert_raises(ValueError, lambda: round_trade_fee(Decimal("-0.000001")))
    print("PASS: upward microdollar trade-fee rounding")


def check_tick_grid():
    valid = ("0.001", "0.005", "0.099", "0.10", "0.11", "0.50", "0.90", "0.901", "0.995", "0.999")
    invalid = ("0", "1", "0.105", "0.505", "0.8995")
    for price in valid:
        assert is_on_tick_grid(price)
    for price in invalid:
        assert not is_on_tick_grid(price)

    generated = all_executable_prices()
    assert len(generated) == len(set(generated)) == 279
    assert sum(is_on_tick_grid(Decimal(mils) / 1000) for mils in range(1, 1000)) == 279
    assert all(is_on_tick_grid(price) for price in generated)
    assert is_on_tick_grid("0.10") and is_on_tick_grid("0.90")

    print("PASS: tapered tick grid (279 executable mil prices)")


def assert_fill_components(result, expected):
    assert isinstance(result, FillFeeResult)
    assert {field.name for field in fields(FillFeeResult)} == set(expected)
    for name, expected_value in expected.items():
        assert getattr(result, name) == expected_value, (name, getattr(result, name), expected_value)


def check_direct_member_single_fills():
    core_buy = calculate_single_fill_fee("0.50", 1, "buy")
    assert_fill_components(core_buy, {
        "executable_price": Decimal("0.50"),
        "contracts": 1,
        "action": "buy",
        "fee_type": KXBTC15M_FEE_TYPE,
        "multiplier": Decimal("1"),
        "rounding_mode": DIRECT_MEMBER,
        "balance_quantum": Decimal("0.0001"),
        "model_fee": Decimal("0.0175"),
        "trade_fee": Decimal("0.017500"),
        "signed_principal_cash_flow": Decimal("-0.50"),
        "pre_rebate_posted_cash_change": Decimal("-0.5175"),
        "rounding_adjustment": Decimal("0"),
        "rebate": Decimal("0"),
        "net_cash_fee": Decimal("0.017500"),
        "posted_cash_change": Decimal("-0.5175"),
        "accumulator_remaining": Decimal("0"),
    })

    tail_buy = calculate_single_fill_fee("0.005", 1, "buy")
    assert_fill_components(tail_buy, {
        "executable_price": Decimal("0.005"),
        "contracts": 1,
        "action": "buy",
        "fee_type": KXBTC15M_FEE_TYPE,
        "multiplier": Decimal("1"),
        "rounding_mode": DIRECT_MEMBER,
        "balance_quantum": Decimal("0.0001"),
        "model_fee": Decimal("0.00034825"),
        "trade_fee": Decimal("0.000349"),
        "signed_principal_cash_flow": Decimal("-0.005"),
        "pre_rebate_posted_cash_change": Decimal("-0.0054"),
        "rounding_adjustment": Decimal("0.000051"),
        "rebate": Decimal("0"),
        "net_cash_fee": Decimal("0.000400"),
        "posted_cash_change": Decimal("-0.0054"),
        "accumulator_remaining": Decimal("0.000051"),
    })

    tail_sell = calculate_single_fill_fee("0.055", 1, "sell")
    assert_fill_components(tail_sell, {
        "executable_price": Decimal("0.055"),
        "contracts": 1,
        "action": "sell",
        "fee_type": KXBTC15M_FEE_TYPE,
        "multiplier": Decimal("1"),
        "rounding_mode": DIRECT_MEMBER,
        "balance_quantum": Decimal("0.0001"),
        "model_fee": Decimal("0.00363825"),
        "trade_fee": Decimal("0.003639"),
        "signed_principal_cash_flow": Decimal("0.055"),
        "pre_rebate_posted_cash_change": Decimal("0.0513"),
        "rounding_adjustment": Decimal("0.000061"),
        "rebate": Decimal("0"),
        "net_cash_fee": Decimal("0.003700"),
        "posted_cash_change": Decimal("0.0513"),
        "accumulator_remaining": Decimal("0.000061"),
    })

    assert core_buy.net_cash_fee != OFFICIAL_PDF_PRINTED_REFERENCE[0][2]
    print("PASS: Direct Member single-fill components (core buy, tail buy, tail sell)")


def check_wrapper_equivalence():
    cases = (
        ("0.005", 1, "buy", Decimal("1"), DIRECT_MEMBER),
        ("0.50", 4, "sell", Decimal("1.5"), DIRECT_MEMBER),
        ("0.055", 1, "buy", Decimal("1"), NON_DIRECT_FCM),
        ("0.90", 25, "sell", Decimal("0"), NON_DIRECT_FCM),
    )
    for price, contracts, action, multiplier, mode in cases:
        accumulator = FeeAccumulator()
        core = calculate_fill_fee(price, contracts, action, accumulator, multiplier=multiplier, rounding_mode=mode)
        wrapped = calculate_single_fill_fee(price, contracts, action, multiplier=multiplier, rounding_mode=mode)
        assert core == wrapped
        assert accumulator.unrebated_rounding == wrapped.accumulator_remaining
        assert accumulator.rounding_mode == mode

    print("PASS: fresh-state wrapper equivalence")


def check_multi_fill_accumulator():
    accumulator = FeeAccumulator()

    start_1 = accumulator.unrebated_rounding
    fill_1 = calculate_fill_fee("0.055", 1, "buy", accumulator)
    assert start_1 == Decimal("0")
    assert fill_1.rounding_adjustment == Decimal("0.000061")
    assert Decimal("0") == Decimal("0.0001") * (Decimal("0.000061") // Decimal("0.0001"))
    assert fill_1.rebate == Decimal("0")
    assert fill_1.accumulator_remaining == Decimal("0.000061")
    assert fill_1.net_cash_fee == Decimal("0.003700")
    assert fill_1.posted_cash_change == Decimal("-0.0587")

    start_2 = accumulator.unrebated_rounding
    fill_2 = calculate_fill_fee("0.055", 1, "buy", accumulator)
    accumulated_before_rebate = Decimal("0.000061") + Decimal("0.000061")
    available_rebate = Decimal("0.0001")
    gross_fill_2_fee = Decimal("0.003639") + Decimal("0.000061")
    assert start_2 == Decimal("0.000061")
    assert fill_2.rounding_adjustment == Decimal("0.000061")
    assert accumulated_before_rebate == Decimal("0.000122")
    assert available_rebate <= accumulated_before_rebate
    assert available_rebate <= gross_fill_2_fee
    assert fill_2.rebate == available_rebate
    assert fill_2.accumulator_remaining == Decimal("0.000022")
    assert fill_2.net_cash_fee == Decimal("0.003600")
    assert fill_2.posted_cash_change == Decimal("-0.0586")
    assert accumulator.unrebated_rounding == fill_2.accumulator_remaining >= 0

    total_cash_change = fill_1.posted_cash_change + fill_2.posted_cash_change
    total_principal = fill_1.signed_principal_cash_flow + fill_2.signed_principal_cash_flow
    total_fees = fill_1.net_cash_fee + fill_2.net_cash_fee
    assert total_cash_change == Decimal("-0.1173")
    assert total_cash_change == total_principal - total_fees
    assert_raises(ValueError, lambda: calculate_fill_fee("0.055", 1, "buy", accumulator, rounding_mode=NON_DIRECT_FCM))

    print("PASS: two-fill Direct Member accumulator and rebate")


def check_fcm_mode():
    direct = calculate_single_fill_fee("0.055", 1, "buy", rounding_mode=DIRECT_MEMBER)
    fcm = calculate_single_fill_fee("0.055", 1, "buy", rounding_mode=NON_DIRECT_FCM)
    assert direct.balance_quantum == DIRECT_MEMBER_BALANCE_QUANTUM
    assert fcm.balance_quantum == NON_DIRECT_FCM_BALANCE_QUANTUM
    assert direct.model_fee == fcm.model_fee == Decimal("0.00363825")
    assert direct.trade_fee == fcm.trade_fee == Decimal("0.003639")
    assert direct.rounding_adjustment == Decimal("0.000061")
    assert fcm.rounding_adjustment == Decimal("0.001361")
    assert direct.net_cash_fee == Decimal("0.003700")
    assert fcm.net_cash_fee == Decimal("0.005000")
    assert direct.posted_cash_change == Decimal("-0.0587")
    assert fcm.posted_cash_change == Decimal("-0.0600")

    state = FeeAccumulator(rounding_mode=NON_DIRECT_FCM)
    calculate_fill_fee("0.055", 1, "buy", state, rounding_mode=NON_DIRECT_FCM)
    assert_raises(ValueError, lambda: calculate_fill_fee("0.055", 1, "buy", state, rounding_mode=DIRECT_MEMBER))
    print("PASS: non-direct/FCM cash rounding and mode isolation")


def check_multiplier_behavior():
    price = Decimal("0.50")
    base = quadratic_model_fee(price, 4, multiplier=1)
    assert quadratic_model_fee(price, 4, multiplier=0) == Decimal("0")
    assert quadratic_model_fee(price, 4, multiplier=1) == Decimal("0.07")
    assert quadratic_model_fee(price, 4, multiplier="1.5") == base * Decimal("1.5")
    assert quadratic_model_fee(price, 4, multiplier=2) == base * 2

    direct_zero = calculate_single_fill_fee("0.055", 1, "buy", multiplier=0)
    fcm_zero = calculate_single_fill_fee("0.055", 1, "buy", multiplier=0, rounding_mode=NON_DIRECT_FCM)
    assert direct_zero.net_cash_fee == Decimal("0")
    assert direct_zero.rounding_adjustment == Decimal("0")
    assert fcm_zero.model_fee == fcm_zero.trade_fee == Decimal("0")
    assert fcm_zero.rounding_adjustment == Decimal("0.005")
    assert fcm_zero.net_cash_fee == Decimal("0.005")
    assert fcm_zero.posted_cash_change == Decimal("-0.06")
    assert fcm_zero.posted_cash_change == fcm_zero.signed_principal_cash_flow - fcm_zero.net_cash_fee

    print("PASS: zero, unit, and non-unit multiplier behavior")


def check_invariants():
    checked = 0
    prices = all_executable_prices()
    contracts_set = (1, 4, 25, 100)
    multipliers = (Decimal("0"), Decimal("1"), Decimal("1.5"))
    for mode in (DIRECT_MEMBER, NON_DIRECT_FCM):
        for action in ("buy", "sell"):
            for multiplier in multipliers:
                for contracts in contracts_set:
                    for price in prices:
                        result = calculate_single_fill_fee(price, contracts, action, multiplier=multiplier, rounding_mode=mode)
                        assert result.posted_cash_change == result.signed_principal_cash_flow - result.net_cash_fee
                        assert result.net_cash_fee >= 0
                        assert result.trade_fee >= result.model_fee
                        assert result.trade_fee - result.model_fee < TRADE_FEE_QUANTUM
                        assert result.rounding_adjustment >= 0
                        assert result.rounding_adjustment < result.balance_quantum
                        assert result.rebate == 0
                        assert result.accumulator_remaining >= 0
                        assert result.accumulator_remaining < result.balance_quantum
                        assert result.rebate <= result.trade_fee + result.rounding_adjustment
                        checked += 1

    print(f"PASS: core accounting invariants ({checked:,} fresh fills)")


def check_guards():
    invalid_prices = (0, 1, -1, Decimal("1.1"), Decimal("NaN"), float("inf"))
    for price in invalid_prices:
        assert_raises((TypeError, ValueError), lambda price=price: quadratic_model_fee(price, 1))
    assert_raises(ValueError, lambda: calculate_single_fill_fee("0.505", 1, "buy"))
    assert calculate_single_fill_fee("0.505", 1, "buy", validate_tick_grid=False).executable_price == Decimal("0.505")

    for contracts in (0, -1, 1.5, Decimal("1.5"), True):
        assert_raises((TypeError, ValueError), lambda contracts=contracts: quadratic_model_fee("0.50", contracts))
    for multiplier in (-1, Decimal("NaN"), float("inf")):
        assert_raises((TypeError, ValueError), lambda multiplier=multiplier: quadratic_model_fee("0.50", 1, multiplier))

    assert_raises(ValueError, lambda: quadratic_model_fee("0.50", 1, fee_type="flat"))
    assert_raises(ValueError, lambda: quadratic_model_fee("0.50", 1, fee_type=7))
    assert_raises(ValueError, lambda: calculate_single_fill_fee("0.50", 1, "hold"))
    assert_raises(ValueError, lambda: calculate_single_fill_fee("0.50", 1, None))
    assert_raises(ValueError, lambda: calculate_single_fill_fee("0.50", 1, "buy", rounding_mode="legacy"))
    assert_raises(TypeError, lambda: calculate_fill_fee("0.50", 1, "buy", object()))

    state = FeeAccumulator()
    state.unrebated_rounding = Decimal("-0.0001")
    assert_raises(ValueError, lambda: calculate_fill_fee("0.50", 1, "buy", state))
    assert_raises(ValueError, lambda: FeeAccumulator(Decimal("-0.0001")))

    bound_state = FeeAccumulator(rounding_mode=DIRECT_MEMBER)
    assert_raises(ValueError, lambda: calculate_fill_fee("0.50", 1, "buy", bound_state, rounding_mode=NON_DIRECT_FCM))

    raw_string = quadratic_model_fee("0.5", 1, multiplier="1")
    assert raw_string == quadratic_model_fee(0.5, 1) == quadratic_model_fee(Decimal("0.5"), 1)
    result_string = calculate_single_fill_fee("0.5", 1, "buy", multiplier="1")
    result_float = calculate_single_fill_fee(0.5, 1, "buy", multiplier=1.0)
    result_decimal = calculate_single_fill_fee(Decimal("0.5"), 1, "buy", multiplier=Decimal("1"))
    assert result_string == result_float == result_decimal

    print("PASS: input guards and equivalent Decimal-compatible inputs")


def main():
    print_official_pdf_reference()
    checks = (
        check_public_api,
        check_raw_model_fee,
        check_float_trap_protection,
        check_trade_fee_rounding,
        check_tick_grid,
        check_direct_member_single_fills,
        check_wrapper_equivalence,
        check_multi_fill_accumulator,
        check_fcm_mode,
        check_multiplier_behavior,
        check_invariants,
        check_guards,
    )
    for check in checks:
        check()
    print(f"\nAll {len(checks)} fee-check groups passed.")


if __name__ == "__main__":
    main()
