# ADR-0008: Reversible PII tokenization at the LLM boundary

**Status:** accepted

**Decision.** Everything that crosses to the model vendor is tokenized (`<EMAIL_1>`, `<PHONE_1>`, `<ADDR_1>`, `<NAME_1>`): user text (regex), tool results (field-aware + regex, scoped per record), and the persisted conversation (checkpoints, traces). Cards/SSNs are hard-redacted and never restorable. Restoration happens only (1) in model-produced tool arguments, because internal systems need real values, and (2) on egress to the customer, **only** for tokens proven to belong to them (their own message, their profile email, or the same record as it). Foreign tokens are shown as inert tokens and logged (`pii_token_withheld`).

**Why measure it.** The eval records every byte sent to the provider and checks for known PII values: 77% of baseline conversations leak, 0% in the final configuration. A control without a metric is a claim.

**Limits.** Detection is structural/regex, not NER: free-form names or addresses typed in prose can slip through. The vault sits in session state and should be encrypted at rest and scoped per tenant. Tokenization can degrade tasks that need the raw value inside the model (rare here; the model needs entities, not contact details).
