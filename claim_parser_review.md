# Claim Parser Review

## Improvements made

- Enhanced customer utterance handling by collecting all customer segments in order, rather than relying only on the final segment.
- Added support for multiple customer utterances to override earlier statements while ignoring support questions.
- Hardened issue-type extraction with priority ordering to select semantically more specific damage classes such as `glass_shatter` over generic matches.
- Removed duplicate keyword entries and reduced redundant patterns in `ISSUE_KEYWORDS`, `PART_KEYWORDS`, and `SEVERITY_KEYWORDS`.
- Added explicit ambiguity pattern matching for normalized phrases, including `i think`, `maybe`, `not sure`, `could be`, `perhaps`, `shayd`, and `sakta hai`.
- Refined severity matching to avoid false positives from generic adjectives like `small` by matching phrase-level low-severity descriptions.
- Added helper methods with docstrings and clearer logging for parser internals.

## Bugs fixed

- Fixed issue extraction where `stain` could override `crack` due to a generic substring match.
- Fixed issue extraction where `missing_part` matched too broadly and could overshadow `dent` in some claims.
- Fixed issue selection ordering for `crushed_packaging` versus `torn_packaging` by using a priority list.
- Fixed severity extraction false positives caused by generic `small` matches.

## Remaining limitations

- The parser still uses simple keyword matching and may fail for highly idiomatic or unseen multilingual expressions.
- It does not resolve claim-object conflicts if the conversation mentions a different object type than `claim_object`.
- It currently returns a single `ClaimTarget` even for multi-part claims, so composite claims are not fully represented.
- Severity extraction is approximate and depends on explicit phrase matches.
- The parser treats all customer utterances as equally weighted; more advanced intent ranking is not yet implemented.

## Future enhancements

- Add part/issue co-occurrence scoring to better resolve conflicting mentions across long conversations.
- Extend support for multi-target claim extraction with multiple `ClaimTarget` entries.
- Add normalization for common code-switched expressions beyond the current keyword list.
- Implement a small rule engine for phrase-based severity classification rather than exact phrase matching.
- Add more unit tests for edge cases such as unsupported objects, mixed-language claims, and compound issue mention sequences.
