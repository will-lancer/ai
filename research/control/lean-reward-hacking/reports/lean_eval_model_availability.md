# Lean-eval model availability

This record fixes the pilot wire IDs and request settings checked on 2026-08-26. It is an availability record, not an authorization to spend. The runner must complete an account-access preflight and a valid, user-issued approval before a generative request is dispatched.

| config key | provider | wire model ID | API and endpoint | request setting |
| --- | --- | --- | --- | --- |
| `openai_sol` | OpenAI | `gpt-5.6-sol` | Responses API, `POST https://api.openai.com/v1/responses` | `reasoning.effort=medium` |
| `openai_luna` | OpenAI | `gpt-5.6-luna` | Responses API, `POST https://api.openai.com/v1/responses` | `reasoning.effort=medium` |
| `anthropic_opus5` | Anthropic | `claude-opus-5` | Messages API, `POST https://api.anthropic.com/v1/messages` | adaptive thinking, `output_config.effort=medium` |
| `google_flash37` | Google | `gemini-3.7-flash` | Gemini API, `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent` | thinking level `medium` |

The pilot treats the config key as a human-readable label and the wire ID as an exact provenance field. Moving aliases, preview IDs, and invented dated suffixes are excluded. The OpenAI `gpt-5.6` alias is excluded even though its current route is Sol. Anthropic's dateless Opus 5 identifier is used as documented; no `-v1` or date suffix is added. Google's stable GA identifier is used instead of `gemini-flash-latest`, preview names, or an invented `001` suffix.

## Provenance captured for every request

The preflight and response record must retain the requested model ID, provider-returned model ID, provider response ID, retrieval date, request settings, package versions, prompt/config hashes, and the hash of the relevant provider documentation. Google preflight also records `models.get` metadata when available, including `name`, `version`, and `baseModelId`. OpenAI pages expose a snapshots section without a dated snapshot string, so the requested ID, returned ID, retrieval date, and documentation hash are the reproducibility boundary. Google's stable model page likewise supplies no dated snapshot string.

Provider adapters keep the model call behind an injected transport. They send no tools, shell capability, filesystem capability, or network capability to the evaluated model. The model receives the task, interface instructions, and sanitized prior Lean feedback only.

## Current documentation and price references

- [OpenAI GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) and [OpenAI GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) document the two requested IDs and available reasoning levels.
- [OpenAI latest-model guidance](https://developers.openai.com/api/docs/guides/latest-model) documents alias and model-selection considerations. [OpenAI pricing](https://developers.openai.com/api/docs/pricing) lists the checked rates: Sol at $4 input and $20 output per million tokens; Luna at $0.20 input and $1.20 output per million tokens.
- [Claude Opus 5 overview](https://platform.claude.com/docs/en/models/opus-5/overview), [Anthropic model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions), and the [Anthropic model overview](https://platform.claude.com/docs/en/models/overview) document the active Opus 5 ID and versioning convention. The checked release date is 2026-07-24, with retirement no sooner than 2027-07-24.
- [Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort) and the [Messages API](https://platform.claude.com/docs/en/api/http/messages/create) document adaptive thinking and `output_config.effort`. Manual `budget_tokens` thinking is not used. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) lists $5 input and $25 output per million tokens for the checked Opus rate.
- [Gemini 3.7 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash), [Gemini model list](https://ai.google.dev/gemini-api/docs/models), [latest-model guidance](https://ai.google.dev/gemini-api/docs/latest-model), and [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking) document the stable ID and medium thinking setting. [Google pricing](https://ai.google.dev/gemini-api/docs/pricing) lists $0.75 input and $3.75 output per million tokens through 2026-12-31, subject to account and tier limits.

Rates and availability can change. The runner logs the retrieval date and fails closed when account access, the requested model ID, the required setting, or approval is unavailable. No paid provider request was made while this report and the offline harness were built.
