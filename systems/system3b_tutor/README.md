System 3B is the adaptive tutor application built on top of the SDK.

Target responsibilities:
- define tutor-specific action bank and state schema
- connect tutor prompts / user responses to System 2 contracts
- run the application loop using System 1 through the SDK runtime

This package is intentionally lightweight right now. Existing tutor code from
`emotiv_learn/` should be migrated here in small, validated steps.
