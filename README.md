# Andromeda Trust Library

Versioned trust-evaluation library consumed by `andromeda-trust` as an external dependency.

## Build a Versioned Package

1. Update `version` in `pyproject.toml`.
2. Build the package:

```bash
./scripts/build-package.sh
```

This generates the distributable artifacts in `dist/`.

## Publish a Wheel for the Trust Microservice

To make the trust microservice consume the library without linking this repository, build the wheel straight into its local wheelhouse:

```bash
./scripts/build-package.sh ../andromeda-trust/.wheels
```

`andromeda-trust` pins the library version in its own `pyproject.toml`, so the wheel version must match that pinned dependency.

If you use an external package index instead of the local wheelhouse, publish the generated artifacts from `dist/` there and keep the same pinned version in `andromeda-trust`.
