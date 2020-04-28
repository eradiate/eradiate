# Eradiate Nested Build Example

## Cloning

```
git clone --recursive <URL>
```

## Setting up and activating Conda env

```
conda env create --file resources/environment.yml
conda activate eradiate_nested
```

## Setting up environment variables

TODO: add this to environment setup (easier).
```
source setpath.sh
```

## Building

Build Mitsuba:
```
cmake . -B build -GNinja
cmake --build build
```

Install the Eradiate package in dev mode:
```
python setup.py develop
```

## Testing

Print embedded Mitsuba version:
```
python -c "import eradiate; eradiate.kernel.set_variant('scalar_rgb'); print(eradiate.kernel.core.MTS_VERSION)"
```
Run the Mitsuba test suite:
```
pytest -m "not slow" ext/mitsuba2/src
```
