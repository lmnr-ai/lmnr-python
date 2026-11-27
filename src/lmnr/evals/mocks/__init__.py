"""`EvalRunner.mocks`: the user-facing handler API (`mock`, re-exported as
`lmnr.evals.mock`), the in-container server (`server`), the host side that
validates, starts and collects it (`host`), and the call log in verify
(`calls`, `ctx.mocks`).

Kept empty on purpose: import the modules by path
(`from lmnr.evals.mocks.host import ...`), so importing one never pulls in the
others and pyright sees no cycle through this file.
"""
