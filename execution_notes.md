# Execution Notes

## 1. Fee Schedule

Research retrieved on **2026-09-14** for KXBTC15M and the evaluation period **2026-05-26 through 2026-08-24**, inclusive. Day 10 Section 1.1 only; no fee implementation or evaluation data was used.

### Published event-contract schedule

The [official fee PDF](https://kalshi.com/docs/kalshi-fee-schedule.pdf), linked directly by the [Help Center](https://help.kalshi.com/en/articles/13823805-fees), is titled *Fee Schedule for July 2026 — 7.7.26 Update* and gives **July 7, 2026** as both its update and effective date. This is not a September effective date.

Its event-contract rule is `fees = round up(M × 0.07 × C × P × (1 − P))`, in dollars. `P` is contract price in dollars; `C` is traded contract quantity; `M` is the applicable multiplier, defaulting to 1 for takers. The exact rounding wording is: “rounds up such that the fee + positionCost is rounded to a centicent”. No separate minimum is specified.

Maker formula: `round up(M × 0.0175 × C × P × (1 − P))`; maker `M` defaults to 0. The non-standard table identifies exceptions, including KXCPI and KXFED. Crypto exceptions include KXBTCMAX150 (maker/taker multipliers 1/1), KXBTCY and KXETHY (0/0); KXBTC15M is absent. Crypto funding fees and perpetual-futures fees are separate products/services.

### Current metadata and maker applicability

The repository's existing production base, `https://external-api.kalshi.com/trade-api/v2`, was retained. An unauthenticated `GET /series/KXBTC15M` returned HTTP 200. The complete body is preserved in [the series snapshot](data/execution/kxbtc15m_series_snapshot.json), with URL and a retrieval timestamp based explicitly on the response's HTTP Date header.

| Field | Fetched value | Meaning |
| --- | --- | --- |
| `fee_type` | `quadratic` | General trading fee structure |
| `fee_multiplier` | `1` | Multiplies the fee calculation without changing its magnitude |
| `last_updated_ts` | `2026-09-08T19:38:35.755394Z` | Metadata update time, not a fee effective date |
| `exchange_index` | `2` | Exchange shard, not a fee multiplier |

The [series schema](https://docs.kalshi.com/api-reference/market/get-series.md) distinguishes `quadratic` from `quadratic_with_maker_fees` and defines `fee_multiplier` as a multiplier on fee calculations. No additional fee fields or fee overrides occur in this response. The raw taker model therefore specializes to **`0.07 × C × P × (1 − P)`**, followed by the rounding treatment below. No maker trading fee is indicated for KXBTC15M, and maker fees do not apply to the specified taker strategy. The taker multiplier of 1 must not be mistaken for an enabled maker multiplier.

### Rounding scope and source discrepancies

The directly retrieved [current Fee Rounding documentation](https://docs.kalshi.com/getting_started/fee_rounding.md) describes fees at **fill level**, with a **per-order accumulator** spanning all fills, including transitions between taker and maker. It does not describe independently rounding each contract or simply rounding an entire arbitrary order once.

For each fill, the documented mechanics are:

1. Round the model fee upward to six decimal dollar places ($0.000001).
2. Floor signed revenue minus that fee to the member's balance precision: **$0.0001 for direct members; $0.01 for non-direct members**. Buy revenue is negative.
3. Treat the difference as a rounding fee and accumulate it within the order.
4. Rebate accumulated overpayment in increments of that balance precision, capped to keep the fill's net fee nonnegative.

Net fee comprises trade fee plus rounding fee minus rebate. These mechanics are documentation, not an implementation in this repository. The current page provides no effective date. Membership/clearing route is not established by the public series response.

The [May 28, 2026 changelog entry](https://docs.kalshi.com/changelog/index.md), titled *Fixed-point dollars added to GET /portfolio/balance*, states that direct-member balances use centicent precision beginning May 28. This date falls inside the evaluation period. It establishes a precision change, but does not provide a complete historical fee-engine specification for every date or clearing route.

Search/browser retrieval initially served an older Fee Rounding page saying all balances were cent-aligned and trade fees were rounded to $0.0001. Direct retrieval of the official `.md` page instead returned the member-specific, six-decimal mechanics above. The live document is recorded here; the older indexed rendering is not treated as current authority.

The PDF's wording and whole-cent example table do not fully explain the newer API mechanics. In particular, its printed $0.02 fee for one $0.50 contract should not be treated as proof of current direct-member rounding. The sources agree on the raw quadratic model but do not establish one universal rounding rule for all users and historical dates. The original Day 10 expectation of upward whole-cent rounding once per order is therefore **not verified as a general rule**.

KXBTC15M uses the quadratic raw trading-fee model `0.07 × C × P(1 − P)`. The exact historical rounding/cash-debit treatment is not yet fully established across the May–August evaluation period and account/clearing routes.

### Official examples and settlement

These are **printed Kalshi table values**, not calculations performed for this project ([PDF, General Trading Fees Table](https://kalshi.com/docs/kalshi-fee-schedule.pdf)):

| P (dollars) | C | Printed fee (dollars) |
| --- | --- | --- |
| 0.50 | 1 | 0.02 |
| 0.50 | 100 | 1.75 |
| 0.40 | 100 | 1.68 |
| 0.10 | 100 | 0.63 |

The current rounding documentation also prints an FCM-cleared example: signed revenue −$0.055000 and supplied model fee $0.00363825 yield trade fee $0.003639, rounding fee $0.001361 and debit $0.06. This illustrates its mechanics, not a KXBTC15M-specific worked example. No fee arithmetic was implemented or independently computed here.

The PDF states no settlement fee. [Current settlement documentation](https://docs.kalshi.com/getting_started/market_settlement.md) confirms zero fees for simple yes/no determinations, but describes sub-cent scalar payout rounding recorded as a settlement fee. Thus zero is supported for ordinary binary settlement, not every exceptional payout. No separate event-contract expiration or redemption charge was found in the reviewed official sources; automatic settlement resolves positions. This is not a claim that broker, withdrawal, or funding charges cannot exist.

The [Help Center Fees article](https://help.kalshi.com/en/articles/13823805-fees), dated April 19, 2026, agrees on immediate executions versus resting orders: maker fees apply only to designated markets when a resting order executes, and canceling a resting order is free. It acknowledges special-market pricing and defers to the PDF. It adds no conflicting formula or rounding prescription. The material discrepancies are between the PDF/table, older indexed API material and current API precision documentation, plus the scalar-settlement qualification above.

### Superseding-schedule search and fee history

Reviewed the [regulatory fee page](https://kalshi.com/regulatory/fee-schedule), [fee-schedule landing page](https://kalshi.com/fee-schedule), Help Center's direct PDF link, current API documentation/changelog, and official-domain searches for June, August and September 2026 schedules. The landing pages exposed little substantive text, but the Help Center link resolved to the July 7 PDF. No later applicable event-contract PDF was found. Older February/October search labels were not accepted as historical versions: reopening the official PDF URLs returned July content. No authenticated archival schedule covering May 26 was recovered.

The changelog contains later changes, including August combo/RFQ maker-fee rules and September margin fee-tier access; neither establishes a KXBTC15M change. The PDF's perpetual-futures section does not supersede the event-contract rules for this series. The search result is bounded evidence, not proof that no unpublished or unindexed update exists.

The [documented series fee-change endpoint](https://docs.kalshi.com/api-reference/exchange/get-series-fee-changes) was called as:

`GET https://external-api.kalshi.com/trade-api/v2/series/fee_changes?series_ticker=KXBTC15M&show_historical=true`

HTTP 200 returned exactly `{"series_fee_change_arr":[]}`: no recorded historical or upcoming series changes, including none in the evaluation window. The September 21, 2025 [changelog](https://docs.kalshi.com/changelog/index) documents that `show_historical=true` requests previous and upcoming changes. The response has no cursor or additional page indicator.

Endpoint functionality was checked with the same parameters for **KXINX**: HTTP 200 returned one record scheduled for **2026-07-03T17:00:00Z**, `fee_type=quadratic`, `fee_multiplier=1`, ID `91b3a604-1798-4dc8-9831-5ecccf5e0758`. Both complete responses, their URLs and server-based retrieval timestamps are preserved in [the fee-change snapshot](data/execution/kxbtc15m_fee_changes_snapshot.json). This proves the endpoint can return historical records, not that its history is exhaustive.

### Historical applicability

**Facts:** The current series has quadratic pricing and multiplier 1. Kalshi returned no series changes with history enabled. The current formal schedule became effective July 7, inside the window. The changelog documents direct-member precision changes on May 28, also inside the window.

**Evidence:** Current metadata, the successful empty history response and the nonempty control response support continuity of the series-specific model and multiplier. They do not establish continuity of exchange-wide rounding. The series metadata update in September cannot independently establish what applied in May.

**Inference:** Using the same raw 0.07 quadratic coefficient and multiplier 1 throughout May 26–August 24 is a reasonable provisional modeling assumption. Applying one exact fee schedule, including rounding, throughout the window is **not established** and is weakened by the dated precision change. The July effective date cannot be moved backward to cover May.

**Historical fee-applicability assumption — the fee schedule used by the backtest is assumed to have applied throughout 2026-05-26 → 2026-08-24.** This is a provisional modeling assumption, not a verified historical fact. Evidence supports the raw KXBTC15M model/multiplier more strongly than rounding continuity. This assumption does not justify silently encoding whole-cent order rounding; historical rounding and the intended direct/non-direct membership route remain unresolved before any implementation is specified.

**Residual uncertainty:** No complete archived pre-July schedule or account-specific historical fee ledger was obtained. Empty series history cannot exclude global fee-engine changes, omissions, or event-level overrides. Kalshi separately documents [event fee overrides](https://docs.kalshi.com/api-reference/events/get-event-fee-changes); individual events were not queried, and absence of their overrides is not claimed. Exact transition timing for historical rounding, applicability to the intended clearing route, and the PDF table/current precision discrepancy remain unresolved. No outcomes, test-split rows, quotes or backtests were read to resolve them.

### Fee implementation decision for Section 1.2

Narrow follow-up verification on **2026-09-14**. The existing snapshots were inspected without refreshing or changing them. This decision supplements the historical findings above; it does not establish a new historical fee record.

**Account route — use Direct Member rounding as the primary model.** Kalshi's [retail deposit instructions](https://help.kalshi.com/en/articles/15426429-how-to-deposit-funds-via-real-time-payments-rtp) explicitly say most users are Direct Members. The [Member Agreement, Sections I–II](https://kalshi.com/docs/kalshi-member-agreement.pdf) covers individuals and incorporates the Kalshi Klear Self-Clearing Member Agreement. Together these support modeling the intended normal retail user trading Predictions directly as a Direct Member. This is a route choice for the project, not verification of any particular person's authenticated account.

In the rounding documentation, non-direct treatment includes the explicitly labeled FCM-cleared example: a customer accesses/clears through a Futures Commission Merchant rather than the direct self-clearing route. Being a retail trader does not itself imply FCM treatment. Broker-intermediated Predictions users can require that alternative, but the stated project model does not. Kalshi's [separate margin-account explanation](https://help.kalshi.com/en/articles/15357608-your-perpetuals-margin-account) distinguishes Predictions balances from perpetual-futures accounts facilitated by Kalshi Prime; Prime/perps status is not the basis for this decision.

**Current mechanics to implement.** The [official Fee Rounding page was retrieved directly as Markdown](https://docs.kalshi.com/getting_started/fee_rounding.md), confirming the six-decimal trade-fee rule, member-specific balance precision and order accumulator. The following mathematical specification is our deterministic translation of that documentation, not code or an additional fee policy. All amounts are dollars. Let `q = 0.0001` for Direct Members, and let `A` be unrebated rounding overpayment belonging only to this order, initially zero.

For each chronological fill:

1. Evaluate the unrounded model fee `f = M × 0.07 × C × P × (1 − P)` for this taker strategy. Preserve exact decimal arithmetic until the documented rounding step; the polynomial itself is not restricted to six decimal places.
2. Set trade fee `t = ceil(f / 0.000001) × 0.000001` (upward to a microdollar).
3. Determine signed, pre-fee cash revenue `r`: `−P × C` for buying the specified contract leg; `+P × C` for selling that leg. The side/price convention must be normalized first, especially for YES-priced API representations of NO trades.
4. Set pre-rebate cash change `b = q × floor((r − t) / q)`, flooring toward negative infinity, including negative buy cash flows. Set rounding fee `u = (r − t) − b`.
5. Add `u` to `A`. Rebate the largest whole multiple of `q` available from `A` that does not exceed this fill's gross fee `t + u`: `v = q × min(floor(A / q), floor((t + u) / q))`. Subtract `v` from `A`; retain any unused overpayment for subsequent fills of this order.
6. Report net cash fee `t + u − v` and posted cash change `b + v` (equivalently `r − net_cash_fee`). Principal and fee remain distinct. Do not additionally round the net fee to whole cents.

The rebate expression makes the documented precision and nonnegative-fee cap explicit. Accumulator state survives across fills and any taker-to-maker transition of the same order; it is not shared between orders. Do not invent an end-of-order refund of a remaining sub-grid balance. Exact decimal/rational arithmetic or suitable fixed-point integers should prevent binary-floating-point ceiling errors.

**Single fill and multiple fills.** For a fresh single-fill order, `A=0` initially and `0 ≤ u < q`, so there is no rebate. If `r` is already on the `q` grid, its net cash fee reduces to rounding the raw model fee upward to `q`, not to $0.01. Otherwise, the fee must also account for principal alignment using signed revenue. This is a derived special case, not a separate rule from Kalshi.

Primary execution assumption: one fresh taker order is represented by one fill at the executable displayed best price for its modeled quantity. This is an explicit abstraction; a displayed price level can match multiple resting orders, so top-of-book price and total quantity alone do not reveal actual fill partitioning. Multiple fills matter because trade-fee rounding is per fill and rebates depend on order history. The fee calculation is deterministic given normalized price/quantity/action, effective fee parameters, rounding mode and ordered fills/initial state. It cannot reconstruct unknown fills from a quote alone. No liquidity, spread or execution-availability analysis is performed here.

**Proposed module structure for Day 11 reuse.** Keep a pure raw quadratic-fee helper separate from cash-rounding helpers. Require an explicit named mode at the cash-fee API boundary: `direct_member` is the project's primary choice; optional `non_direct_fcm` uses the same documented mechanics with `q=0.01`. Provide a single-fill convenience path that initializes state to zero, and an order/fill path that carries state and returns model fee, trade fee, rounding adjustment, rebate, net fee and cash change. Do not return raw fee under a cash-fee name. Reject invalid inputs and unsupported fee types rather than silently falling back to quadratic pricing. The pure fee module should not fetch metadata or infer the clearing route.

The original whole-cent `ceil_to_cent()` model is **rejected for primary execution**. It may be retained only as an explicitly labeled `legacy_whole_cent_sensitivity` approximation, with no claim that it reconstructs historical fees. It is not synonymous with the full FCM path: principal alignment and fill/accumulator state still matter there. No sensitivity calculation is requested or performed now.

**Event override support.** The [current event fee-change schema](https://docs.kalshi.com/api-reference/events/get-event-fee-changes.md) confirms `GET /events/fee_changes`, optional `event_ticker`, and paginated results containing `fee_type_override`, `fee_multiplier_override` and `scheduled_ts`. Overrides sit above the parent series; null fields revert to the corresponding series setting. The module should accept externally resolved effective fee type and multiplier (including zero), allowing an event-specific setting to be supplied later. Do not multiply a replacement event multiplier by the series multiplier. Time-aware override resolution and source provenance belong to the caller/configuration layer; unknown fee types must fail explicitly until implemented. No historical events were queried, and no absence of overrides is inferred.

**Facts, assumptions and remaining blocker.** Direct-member precision and the current mechanics are documented facts; selecting this route and the single-fill abstraction follows the user's intended primary execution model. Applying these current mechanics over all of **2026-05-26 → 2026-08-24** remains the provisional historical fee-applicability assumption, not a historical verification. The existing May 28 precision-change evidence and pre-July/document-table uncertainty remain in force. Do not silently infer or switch to a historical rule on May 26–27; any dated schedule reconstruction would need separate evidence. No unresolved account-route or arithmetic-design issue prevents beginning Section 1.2 for this explicitly labeled model. Exact historical/account-ledger replication remains unproven. This follow-up specifies the implementation only; it does not begin it.

## 2. Edge and Execution Definitions

Day 10/11 uses one primary trading probability: `p` is the train-fitted Platt calibration applied to `p_5min_ewma_vol`, separately by horizon. Only parameter rows with `fit_split == "train"` and `parameter_role == "legitimate_train_fit"` may be used. Raw Stage 0 probabilities, validation-fitted ceiling parameters, refitted parameters and newly selected sigma estimates are not the primary trading probability.

For every row, define `bid = quote_yes_bid`, `ask = quote_yes_ask` and `mid = quote_mid`. The executable price for buying YES is `ask`; the executable price for buying NO is `1 - bid`. Also define:

`disagreement = p - mid`

`half_spread = (ask - bid) / 2`

The five conceptual layers are:

1. **Predictive disagreement.** `p - quote_mid` establishes only that the model and market differ. It is not a tradeable edge.
2. **Gross executable edge.** After crossing the quoted spread but before fees, `gross_edge_yes = p - ask` and `gross_edge_no = (1 - p) - (1 - bid) = bid - p`.
3. **Estimated/net trading edge.** Expected profit per contract if `p` were the true probability, after quoted execution cost and fees, is `net_edge_yes = p - ask - fee_yes` and `net_edge_no = bid - p - fee_no`. Here `fee_yes` is the per-contract fee for buying YES at `ask`, and `fee_no` is the per-contract fee for buying NO at `1 - bid`.
4. **Threshold-clearing signal.** A net edge that exceeds the later frozen basis-risk plus model-error threshold is a candidate trade. Clearing that threshold does not establish a validated or proven edge.
5. **Realized/validated edge.** This is the outcome-based P&L of threshold-clearing signals. Realized edge and P&L are not computed in Day 10 Section 2.1.

The executable-edge formulas have the equivalent midpoint-scale identities:

`gross_edge_yes = disagreement - half_spread`

`gross_edge_no = -disagreement - half_spread`

**No-double-counting rule:** because gross and net edge are measured against executable bid/ask prices, the spread has already been crossed. Do not subtract half-spread again from an ask/bid-based executable edge. The half-spread term is used only in the equivalent midpoint-scale representation or reporting.

**YES/NO exclusivity invariant:** positive gross YES edge requires `p > ask`, while positive gross NO edge requires `bid > p`. Both conditions cannot hold simultaneously unless `bid > ask`, which is an invalid/crossed book. Therefore, on a valid row, at most one side can have positive gross executable edge. This becomes an assertion in later implementation.

### Day 9 Platt probability reproduction check

The carry-forward verification used the existing `apply_platt(...)` helper, `p_5min_ewma_vol`, and only the two stored `legitimate_train_fit` parameter rows, keyed separately by `horizon_minutes`. The rows in `data/models/stage0_model_market_gap.parquet` were matched to `data/models/stage0_predictions.parquet` on the established unique row key `(ticker, horizon_minutes)` plus the carried `split`; no loose matching was used. All 12,883 train/validation rows in the model-market-gap artifact were checked, with no test rows or settlement outcomes read. The raw carried probability also matched `raw_model_probability` exactly.

Recomputed train-fitted Platt probabilities matched `platt_model_probability` with maximum absolute difference `0`, satisfying the required tolerance of `1e-12`. **PASS.** No validation-fitted ceiling parameter was used, no artifact was modified or created, and no fitting or sigma selection was performed.

## 3. Execution Assumptions

All historical trading results produced under this register are **Tier 2 — quote-aware, top-of-book, size-unaware**. The historical data supplies usable quoted bid/ask prices, but it does not establish displayed size, order-book depth, queue position, actual latency or realized slippage for live orders. The ten assumptions below apply together to every later historical backtest unless a separately labeled result tier explicitly replaces them.

| ID | Assumption | Current evidence | Known limitation | Future validation or improvement |
| --- | --- | --- | --- | --- |
| **A1** | **Execution price.** Buying YES executes at `quote_yes_ask`; buying NO executes at `1 - quote_yes_bid`. The price is the stored decision-minute quote. Midpoint execution is prohibited. | Stored decision-time bid/ask fields and the executable-edge definitions frozen in Section 2.1 support this convention. | A stored quote does not prove that a live order would receive that price. | Compare submitted and filled prices against contemporaneous live quote snapshots. |
| **A2** | **Top-of-book fillability.** One contract is assumed fillable at the displayed top-of-book executable price. | The historical dataset records top-of-book prices. | This is not historically proven: historical displayed size and depth are unavailable. Queue position and available quantity are also unknown. | Collect live depth/order-book data and verify one-contract fillability at decision time. |
| **A3** | **No latency.** Decision and execution are assumed to occur at the same quoted price, with no additional latency slippage. | Decision timestamps and their stored quotes define a reproducible simultaneous-execution abstraction. | This is an optimistic simplification; actual latency and price movement before fill are unknown. | Use live order, quote and depth timestamps to estimate latency and run pre-registered latency sensitivities. |
| **A4** | **Taker-only execution.** Every simulated entry crosses the spread as a taker order. There are no maker fills, resting limit orders, queue-position models, or maker rebate/fee assumptions. | Ask/bid execution and the existing taker-fee calculations are aligned with this rule. Crossing the spread is conservative relative to assuming unproven maker fills. | It does not represent strategies that rest orders or earn maker economics. | Evaluate maker execution only after live queue/fill data can support a separate execution model. |
| **A5** | **Historical fee applicability.** Use the current frozen KXBTC15M quadratic mechanics and multiplier throughout the historical window: raw fee `0.07 × C × P × (1 - P)`, followed by the frozen Direct Member cash-rounding model. | Section 1 documents current series multiplier `1`, no returned series fee changes, and the implemented `calculate_single_fill_fee(...)` mechanics. Raw-model and multiplier continuity are better supported than cash-rounding continuity. | Exact historical account-route and cash-rounding continuity across May–August are not fully established; this is not a proven historical fact. | Obtain archived schedules, account-route evidence, or historical fill/ledger records and label any resulting reconstruction separately. |
| **A6** | **Stale-quote exclusion.** A row with `quote_age_seconds > 60` is signal-ineligible. The row remains in the historical dataset; it simply cannot generate a trading signal. Section 2.2 did not remove stale rows. | Quote age is observed, and Section 2.2 found only the small counts recorded below. The rule is frozen before signal-level P&L inspection. | Sixty seconds is a practical cutoff, not a uniquely optimal or empirically proven fillability boundary. | Test against live quote continuity and, after the frozen evaluation, use separately pre-registered sensitivity cutoffs. |
| **A7** | **Fixed order size.** Every simulated trade is exactly `1 contract`; there is no scaling or position sizing. | One contract minimizes dependence on unavailable historical liquidity and size. It also matches the existing single-contract fee analysis. | Even one-contract fillability is assumed, and the results do not establish capacity. | Use live displayed depth and realized fills before introducing size or capacity models. |
| **A8** | **Hold to settlement.** Once entered, the contract is held to settlement, with no early exit, second spread crossing or exit trading fee. Day 11 P&L therefore models only entry execution, entry fee and settlement payoff. | Binary KXBTC15M contracts have a defined settlement payoff, and this rule avoids unsupported exit execution assumptions. | It does not measure stop-losses, profit-taking, inventory management or tradable exit prices. | A separate strategy may model exits after suitable intramarket quotes and execution data are available. |
| **A9** | **Settlement fee.** No ordinary settlement fee is modeled for standard binary KXBTC15M settlement. | Section 1 found no normal separate binary settlement fee for this strategy. | Exceptional settlement-rounding behavior may exist in other contexts; this assumption is limited to ordinary binary settlement. | Reconcile against applicable rule changes and account ledger entries if exceptional settlement mechanics become relevant. |
| **A10** | **Settlement and basis handling.** Kalshi's official settlement result/final settlement value is accepted as contract truth. The external BTC proxy never overwrites Kalshi settlement. Measured BTC/Kalshi basis uncertainty is handled later through the uncertainty/threshold framework, not by altering outcomes. | Kalshi determines the contract payoff, while Day 9 established that external-exchange basis risk is material. | The proxy may differ from Kalshi's settlement source, and this section does not construct or validate any threshold. | Validate the later uncertainty framework independently while preserving official Kalshi settlement as the outcome of record. |

### Known historical fields versus execution assumptions

The historically known/measured fields are bid, ask, quote age, spread and timestamps. One-contract top-of-book fillability, available size, order-book depth, queue position, latency and live slippage are assumed or unavailable—not historical facts. That boundary is why current results must remain labeled Tier 2.

### Frozen stale-quote policy

The `quote_age_seconds > 60` signal-ineligibility rule is frozen now because quote age is an observable data-quality property and the rule can be chosen before inspecting signal outcomes or P&L. It prevents treating a quote more than one minute old as necessarily executable while retaining the row in the underlying data. Sixty seconds is a practical frozen assumption, not a claim of unique optimality.

Section 2.2 measured the following all-row counts; the Day 9 common-population counts were identical:

| Split | Horizon | Rows with `quote_age_seconds > 10` | Rows with `quote_age_seconds > 60` |
| --- | --- | ---: | ---: |
| Train | T-10 | 2 | 2 |
| Train | T-5 | 3 | 3 |
| Validation | T-10 | 0 | 0 |
| Validation | T-5 | 1 | 0 |

No row is removed by this documentation step. No basis/model-error threshold, edge, signal-clearance result, settlement evaluation or P&L is computed here.

## 4. Measured Spreads and Fee Exposure

All price-aware findings in this section are **Tier 2 — quote-aware, top-of-book, size-unaware**. They describe stored train/validation quotes and the frozen one-contract Direct Member fee model; they do not establish historical fillability, depth, latency, slippage or profitability. `data/execution/spread_summary.parquet` contains 256 summary rows for both the all-row market population and the Day 9 common population, split by overall, UTC hour and the seven `quote_mid` price buckets. Test rows and outcome columns were not loaded.

### Overall spread structure

Spreads are integer mils, where 1 mil is `0.001` probability units. The all-row split-level distributions were:

| Split | Horizon | Rows | Mean | Median | p10 | p25 | p75 | p90 | p95 | p99 | Max | Share 1 mil | Share 10 mils | Share >20 mils |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | T-10 | 5,196 | 10.149 | 10 | 10 | 10 | 10 | 10 | 20 | 20 | 140 | 4.27% | 85.62% | 0.25% |
| Train | T-5 | 5,196 | 7.158 | 10 | 1 | 2 | 10 | 10 | 13 | 20 | 220 | 21.86% | 48.34% | 0.29% |
| Validation | T-10 | 1,983 | 9.527 | 10 | 10 | 10 | 10 | 10 | 10 | 20 | 20 | 5.90% | 90.47% | 0.00% |
| Validation | T-5 | 1,983 | 6.174 | 10 | 1 | 1 | 10 | 10 | 10 | 20 | 30 | 33.64% | 52.14% | 0.05% |

The Day 9 common population was reproduced exactly and remained close to the all-row distribution:

| Split | Horizon | Common rows | Mean spread (mils) | Median spread (mils) |
| --- | --- | ---: | ---: | ---: |
| Train | T-10 | 4,792 | 10.130 | 10 |
| Train | T-5 | 4,757 | 7.087 | 10 |
| Validation | T-10 | 1,685 | 9.535 | 10 |
| Validation | T-5 | 1,649 | 6.082 | 10 |

### Price region dominates the spread pattern

The strong result is the difference between the tapered tails (`p00_10` and `p90_100`) and the core (`p40_60`).

| Population | Split | Horizon | Tail mean spread (mils) | Core mean spread (mils) | Absolute difference (mils) |
| --- | --- | --- | ---: | ---: | ---: |
| All rows | Train | T-10 | 2.378 | 10.806 | 8.428 |
| All rows | Train | T-5 | 2.686 | 11.039 | 8.353 |
| All rows | Validation | T-10 | 1.461 | 10.223 | 8.763 |
| All rows | Validation | T-5 | 1.557 | 10.225 | 8.668 |
| Day 9 common | Train | T-10 | 2.343 | 10.777 | 8.434 |
| Day 9 common | Train | T-5 | 2.651 | 11.030 | 8.379 |
| Day 9 common | Validation | T-10 | 1.477 | 10.249 | 8.772 |
| Day 9 common | Validation | T-5 | 1.553 | 10.242 | 8.689 |

On validation common rows, the two tail buckets had mean spreads of `1.352` and `1.564` mils at T-10 and `1.577` and `1.528` mils at T-5, with 1-mil medians. The five interior buckets were approximately 10 mils: their means ranged from `10.046` to `10.327` mils at T-10 and `10.136` to `10.340` mils at T-5, with 10-mil medians throughout. Price region is therefore materially more informative about quoted spread than horizon alone.

UTC hour was a weak result relative to price region. The maximum within-population/split/horizon range of hourly mean spreads was `2.556` mils, versus tail/core differences of roughly `8.35–8.77` mils. The hourly pattern was not perfectly flat—the T-5 ranges crossed the script's descriptive 2-mil flag—but it did not approach the price-region effect and is treated as effectively a null/secondary finding rather than an execution-time rule.

### Direct Member fee exposure and midpoint-scale entry cost

Fees below are the actual `net_cash_fee` returned by `calculate_single_fill_fee(...)` for a fresh one-contract Direct Member taker buy at the executable price. They are not raw quadratic fees and are not a hard-coded 1¢/2¢ classification.

| Split | Horizon | YES fee mean | YES fee median | NO fee mean | NO fee median | YES midpoint-cost median | NO midpoint-cost median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | T-10 | $0.013379 | $0.014700 | $0.013377 | $0.014700 | 0.0200 | 0.0200 |
| Train | T-5 | $0.008222 | $0.006900 | $0.008215 | $0.006900 | 0.0124 | 0.0124 |
| Validation | T-10 | $0.013535 | $0.015000 | $0.013523 | $0.015000 | 0.0200 | 0.0200 |
| Validation | T-5 | $0.008289 | $0.007400 | $0.008277 | $0.006900 | 0.0124 | 0.0122 |

Fee amounts are dollars per contract and therefore numerically equal probability units for a binary one-dollar payoff. Midpoint-cost values are `half_spread + fee` in probability units and are reporting quantities only. They show how much model-versus-mid disagreement is consumed before executable net edge becomes positive. They must never be subtracted from an ask/bid-based edge a second time.

Across all-row overall groups, one-contract net cash fees ranged from `$0.0001` in the deepest T-5 tails to `$0.0175` at the top of the quadratic schedule; T-10 minima ranged from `$0.0007` to `$0.0012` depending on side and split. The validation-common core (`p40_60`) median fee was `$0.0174` on both sides, versus a 5-mil half-spread, so the fee was the larger quoted entry-cost component. In the validation-common tails, median fees were roughly `$0.0023–$0.0053` against a roughly 0.5–0.8 mil half-spread, again making fees larger than the spread component. The original plan's obsolete 1¢/2¢ fee-share classification was intentionally not separately materialized.

### Tapered ticks and quote age

Sub-cent quote shares on all rows were:

| Split | Horizon | Bid sub-cent share | Ask sub-cent share |
| --- | --- | ---: | ---: |
| Train | T-10 | 6.66% | 6.76% |
| Train | T-5 | 41.11% | 41.49% |
| Validation | T-10 | 6.81% | 6.81% |
| Validation | T-5 | 41.60% | 42.41% |

The effect is concentrated in the tapered tails. On validation common rows, tail-bucket sub-cent shares were approximately `88–93%`, while core buckets were zero and nearly all interior buckets were zero. This supports using exact mil/tick-aware handling rather than string or binary-float precision tests.

Stale rows were retained in Section 2.2. All-row counts were 2/2 (`>10s`/`>60s`) for train T-10, 3/3 for train T-5, 0/0 for validation T-10 and 1/0 for validation T-5. Day 9 common-population counts were identical. These observations support A6's pre-outcome rule that rows older than 60 seconds remain in the denominator but are signal-ineligible.

## 5. Frozen Edge Threshold

The authoritative composition rule is the pre-registered **Day 10 Edge Threshold Rule** in `evaluation_notes.md`. It applies to net executable edge:

`net_edge_yes = p - ask - fee_yes`

`net_edge_no = bid - p - fee_no`

No half-spread is subtracted again because the ask/bid executable price already crosses the spread. The trading probability is the horizon-specific train-fitted Platt calibration of `p_5min_ewma_vol`; order size is one contract. Raw probability and validation-fitted ceiling parameters do not define the shipping threshold.

Rows use the seven frozen `quote_mid` buckets `p00_10`, `p10_25`, `p25_40`, `p40_60`, `p60_75`, `p75_90` and `p90_100`. The primary basis size is `1.2 bps`; `5.0 bps` is a conservative sensitivity, not primary. For each horizon/bucket, `basis_term` is validation median `max_abs_shift` for the train-fitted Platt probability. The horizon-wide `model_error_term` is 10-decile validation ECE. The combination is additive:

`required_net_edge = basis_term + model_error_term`

The ECE terms are `0.035595749` at T-10 and `0.021091839` at T-5. With approximately 165 rows per decile, binomial sampling noise near `p = 0.5` is roughly 3–4 percentage points per bin, so these ECE margins have a known upward bias. They were not bias-corrected, replaced or tuned after inspection. The diagnostic maximum absolute decile deviations were `0.073602229` at T-10 and `0.052676878` at T-5; they do not enter the threshold.

### Primary threshold rows

All quantities except row counts and `basis_bps` are raw probability units. These are the complete 14 primary rows in `data/execution/edge_threshold.parquet`:

| Horizon | Price bucket | Validation rows | Basis term | Model-error term | Required net edge |
| --- | --- | ---: | ---: | ---: | ---: |
| T-10 | `p00_10` | 54 | 0.010026765 | 0.035595749 | 0.045622514 |
| T-10 | `p10_25` | 279 | 0.030700948 | 0.035595749 | 0.066296697 |
| T-10 | `p25_40` | 317 | 0.049957631 | 0.035595749 | 0.085553379 |
| T-10 | `p40_60` | 442 | 0.052070246 | 0.035595749 | 0.087665994 |
| T-10 | `p60_75` | 275 | 0.045566593 | 0.035595749 | 0.081162342 |
| T-10 | `p75_90` | 240 | 0.029201283 | 0.035595749 | 0.064797031 |
| T-10 | `p90_100` | 78 | 0.014255799 | 0.035595749 | 0.049851548 |
| T-5 | `p00_10` | 404 | 0.009467124 | 0.021091839 | 0.030558963 |
| T-5 | `p10_25` | 217 | 0.054848685 | 0.021091839 | 0.075940524 |
| T-5 | `p25_40` | 147 | 0.080369751 | 0.021091839 | 0.101461591 |
| T-5 | `p40_60` | 165 | 0.085942916 | 0.021091839 | 0.107034755 |
| T-5 | `p60_75` | 147 | 0.082033544 | 0.021091839 | 0.103125383 |
| T-5 | `p75_90` | 183 | 0.052249659 | 0.021091839 | 0.073341498 |
| T-5 | `p90_100` | 386 | 0.011468007 | 0.021091839 | 0.032559846 |

The primary required-net-edge range is `0.045622514–0.087665994` at T-10 and `0.030558963–0.107034755` at T-5. No sparse-bucket merge fired.

The 14 conservative-sensitivity rows use the same buckets, ECE and additive rule with `basis_bps = 5.0`. Their basis terms range from `0.056761151–0.230252825` at T-10 and `0.077308148–0.344941873` at T-5. Their required-net-edge ranges are `0.092356900–0.265848574` at T-10 and `0.098399988–0.366033712` at T-5.

For context, Day 9's validation median absolute calibrated model-market disagreement was approximately 5.23 percentage points at T-10 and 3.84 percentage points at T-5. Those values are on the midpoint-disagreement layer; disagreement itself was not executable edge and did not account for spread crossing, fees or the uncertainty threshold. The threshold was computed once from the saved rule and was not adjusted after Section 3.3 observed clearance.

## 6. Outcome-Free Threshold Clearance

**Tier 2 inputs, outcome-free: candidate signals, not trades, not P&L.** This label applies to every result in this section. Section 3.3 used train and validation common rows only, retained stale rows in the denominator, applied A6 eligibility, and wrote summary rows only.

### Primary train-fitted Platt waterfall

| Split | Horizon | Common | Stale excluded | Gross-positive best side | Net-positive best side | Threshold clear |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | T-10 | 4,792 | 2 | 4,517 | 3,847 | 890 |
| Train | T-5 | 4,757 | 3 | 4,583 | 4,244 | 1,357 |
| Validation | T-10 | 1,685 | 0 | 1,597 | 1,369 | 368 |
| Validation | T-5 | 1,649 | 0 | 1,599 | 1,480 | 460 |

### Raw diagnostic versus train-fitted Platt

| Split | Horizon | Raw, 1.2 bps | Platt, 1.2 bps | Raw, 5.0 bps | Platt, 5.0 bps |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | T-10 | 2,370 | 890 | 650 | 81 |
| Train | T-5 | 3,486 | 1,357 | 1,628 | 217 |
| Validation | T-10 | 778 | 368 | 216 | 31 |
| Validation | T-5 | 1,153 | 460 | 543 | 73 |

Raw results are diagnostic only; train-fitted Platt remains the shipping probability. `1.2 bps` remains primary and `5.0 bps` remains the conservative sensitivity. The threshold was not changed after these counts were observed.

### Primary Platt side and clearing-edge summaries

| Split | Horizon | YES clears | NO clears | Median best-side net edge | p90 best-side net edge |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | T-10 | 346 | 544 | 0.102171108 | 0.165530795 |
| Train | T-5 | 671 | 686 | 0.101601873 | 0.202556506 |
| Validation | T-10 | 106 | 262 | 0.107377628 | 0.172583560 |
| Validation | T-5 | 171 | 289 | 0.104996412 | 0.212260112 |

Net-edge statistics are raw probability units and apply only to threshold-clearing rows. They are estimated model-based margins, not returns or realized gains. There were no exact best-side ties.

### Primary Platt distinct-market diagnostics

| Split | Any horizon clears | Both horizons clear | Same side | Opposite side | YES/YES | NO/NO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,881 | 366 | 312 | 54 | 151 | 161 |
| Validation | 719 | 109 | 86 | 23 | 21 | 65 |

These are descriptive market-level diagnostics only. T-10 and T-5 share a target, but Day 10 did not collapse them into a one-position-per-market trading decision. No clearance count is interpreted as profitability, and no candidate's settlement result was inspected.

## 7. What This Does and Does Not Establish

Day 10 maintains five distinct layers:

1. **Predictive disagreement:** `p - quote_mid`; evidence that model and market differ, not an edge.
2. **Gross executable edge:** `p - ask` for YES or `bid - p` for NO; spread crossed, before fees.
3. **Estimated/net trading edge:** gross executable edge minus the side's Direct Member fee; expected profit per contract only if `p` were the true probability.
4. **Threshold-clearing signal:** signal-eligible best-side net edge at least as large as the frozen basis-plus-model-error threshold; a candidate, not a trade or validated edge.
5. **Realized/validated edge:** outcome-based result after applying a frozen trading decision and accounting rule.

Day 10 establishes only layers 1–4. No row has yet been shown to have realized edge, no candidate has yet been shown to win, and no profitability claim is permitted. Day 11 is the first outcome-based, quote-aware evaluation.

The current evidence remains Tier 2: historical top-of-book prices are observed, but size and depth are not. One-contract top-of-book fillability is assumed; latency is zero by assumption; live slippage, depth and queue position are unavailable; all entries are taker-only; and historical fee applicability remains the qualified assumption documented in A5. These limitations apply even when a candidate clears the uncertainty threshold.

## 8. What Day 10 Did Not Do

Day 10 performed:

- no P&L calculation;
- no hit-rate or win-rate calculation;
- no return calculation;
- no outcome join to a candidate signal;
- no Day 11 one-position-per-market trading decision;
- no position sizing or scaling;
- no test-split read of any kind;
- no sigma reselection;
- no Platt refit;
- no use of validation-fitted ceiling parameters as a shipping probability;
- no threshold adjustment after observing clearance;
- no maker execution model; and
- no size-aware depth or slippage model.

Day 10 closes with a frozen probability, executable-edge definition, fee model, execution-assumption register, uncertainty threshold and outcome-free candidate-signal summary. It does not close the gap between estimated edge and realized edge; that outcome-based question begins on Day 11, not here.
