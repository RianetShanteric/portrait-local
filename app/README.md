# App

Desktop UI, user actions, job orchestration and preview state belong here.

The UI may depend on public interfaces from `core` and `pipeline`; it must not
import concrete model implementations from `providers` directly.
