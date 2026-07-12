# CaSiNo Strategy Protocol v1

Status: **APPROVED for protocol semantics**  
Authority: CaSiNo paper, Section 3 “Strategy Annotations” and Table 2, with the project owner's decisions recorded on 2026-07-12.  
Canonical numeric IDs are fixed by `configs/strategy_label_space.json`.

This protocol governs future counterfactual generation, human review, Judge A, Judge B, preference pairs, M-series training, and evaluation. Historical annotations and generations are immutable evidence and must not be rewritten.

## 0. `elicit-pref`

- 中文解释：尝试发现谈判对手对物品的偏好顺序或优先级。
- English definition: Attempts to discover the negotiation partner's preference order over the negotiated items.
- Positive criteria: asks which item the partner prefers, ranks highest, values most, or treats as the highest priority.
- Negative criteria: a generic question, social question, or request for a reason that does not function to discover item preference/order.
- Common confusions: `small-talk`, `other-need`, `promote-coordination`.
- Positive example: “Which of the three supplies is most important to you?”
- Counterexample: “Why are you going camping?” (generic context question, not preference-order elicitation).

## 1. `no-need`

- 中文解释：当前发言人依据自己的 personal context 表示自己对某项物品不需要、需求较低或已有足够供应。
- English definition: The current speaker states, based on personal context, that they do not need, have low need for, or already have enough of an item.
- Positive criteria: self-directed low/no need or sufficient existing supply for a concrete item.
- Negative criteria: merely conceding/giving an item without stating low need or sufficient supply.
- Common confusions: `uv-part` (partner-directed), `promote-coordination` (trade/concession).
- Positive example: “I already brought enough food, so I do not need much more.”
- Counterexample: “You can have two food packs.” (concession alone).

## 2. `other-need`

- 中文解释：当前发言人为自己之外、但与自己相关的第三方建立具体物品需求，例如孩子、家庭、朋友、团队成员或同行者。
- English definition: Similar to Self-Need, but used when the current speaker establishes an item need for someone other than themselves, such as their children, family, friends, group members, or camping companions.
- Positive criteria: the speaker asserts that an associated third party needs a negotiated item or provides that third party's reason for needing it.
- Negative criteria: acknowledging the negotiation partner's need; asking about the partner's preference; stating only the speaker's own personal need.
- Common confusions: `self-need`, `showing-empathy`, `elicit-pref`.
- Positive example: “My kids need more food for the trip.”
- Counterexamples: “I personally need more food.” (`self-need`); “You said your kids need more food.” (not automatically `other-need`); “Do your kids need more food?” (usually `elicit-pref` if discovering preference/priority).

## 3. `promote-coordination`

- 中文解释：当前发言人推动双方协调、交换、互相让步或共同达成交易。
- English definition: The current speaker promotes coordination through an explicit trade offer, mutual concession, exchange, or joint effort to find a deal.
- Positive criteria: explicit give/get trade, mutual concession, concrete exchange, or explicit joint deal-seeking.
- Negative criteria: a fairness/imbalance complaint without a coordination move; a unilateral allocation demand.
- Common confusions: `vouch-fair`, non-strategic allocation statements.
- Positive example: “If you give me two water packs, I can give you all three firewood packs.”
- Counterexample: “That split leaves me with nothing.” (`vouch-fair` when functioning as an imbalance callout).

## 4. `self-need`

- 中文解释：当前发言人为自己建立某项物品的个人需求或需求理由。
- English definition: The current speaker establishes their own personal need for an item or gives a personal reason for needing it.
- Positive criteria: self-directed item need/reason.
- Negative criteria: need attributed only to children/family/group members; partner-directed need acknowledgment.
- Common confusions: `other-need`, historical incorrect “unique value” interpretation of `uv-part`.
- Positive example: “I need the water because I become dehydrated quickly.”
- Counterexample: “My children need the food.” (`other-need`).

## 5. `showing-empathy`

- 中文解释：对谈判对手的 personal context 进行积极承认、理解或表达同理心。
- English definition: Positively acknowledges or expresses empathy toward the negotiation partner's personal context.
- Positive criteria: a clear positive acknowledgment tied to the partner's personal situation, considering context and response together.
- Negative criteria: an isolated formulaic “I understand” with no clear connection to partner personal context; merely stating the partner's allocation preference.
- Common confusions: `other-need`, generic politeness, `small-talk`.
- Positive example: “I understand why keeping your children warm is important to you.”
- Counterexample for first counterfactual batch: “I understand.”

## 6. `small-talk`

- 中文解释：讨论谈判和物品分配之外的社交性内容，以建立 rapport。
- English definition: Social content outside negotiation and item allocation, used to build rapport.
- Positive criteria: greeting, social exchange, camping-related social conversation not used to negotiate item preferences.
- Negative criteria: substantive preference, need, allocation, trade, or fairness discussion.
- Common confusions: `showing-empathy`, generic questions versus `elicit-pref`.
- Positive example: “I hope you have a great camping trip!”
- Counterexample: “Which item is your highest priority?” (`elicit-pref`).

## 7. `uv-part` — Undervalue-Partner

- 中文解释：当前发言人削弱、贬低或质疑谈判对手对某项物品的需求或需求强度。
- English definition: The current speaker undervalues or questions the negotiation partner's need or need strength for an item.
- Positive criteria: says the partner already has basic/enough supplies, does not need additional quantity, has an alternative way to obtain the item, or otherwise weakens the partner's necessity claim.
- Direction rule: the behavior must semantically target the negotiation partner or the partner's requirement, but literal “you” is not required.
- Context rule: “You already have enough” is valid only when context resolves the concrete item and the function is to weaken the partner's need.
- Negative criteria: explaining why the item is uniquely valuable to the current speaker; current speaker's own low need.
- Common confusions: `no-need` (speaker's own low need), `self-need` (speaker's own need reason).
- Positive examples: “You already have the basic supplies, so you probably do not need more firewood.”; “There may be a store nearby where you can get more supplies.”
- Counterexample: “Water is uniquely important to me because of my medical condition.” (`self-need`).

## 8. `vouch-fair`

- 中文解释：为了个人利益诉诸公平，指出方案公平、不公平、偏向对方或使自己明显受损。
- English definition: Appeals to fairness for personal benefit by characterizing a proposal as fair/unfair or calling out an allocation imbalance.
- Positive criteria: explicit fair/unfair statement; acceptance of a fair arrangement; explicit or implicit imbalance callout, including that the speaker receives nothing or the proposal strongly favors the partner.
- Negative criteria: a trade, compromise, or mutual concession without a fairness/imbalance callout.
- Common confusions: `promote-coordination`.
- Positive example: “That proposal gives you nearly everything and leaves me with nothing.”
- Counterexample: “I can give you food if you give me water.” (`promote-coordination`).

## First counterfactual-batch strictness

The first batch accepts only behaviorally clear positives. Context-dependent borderline examples may remain eligible for final multilabel evaluation but must not be used as clear synthetic positives. In particular, isolated “I understand,” concession-only `no-need`, generic questions, and compromise-only `vouch-fair` are rejected.
