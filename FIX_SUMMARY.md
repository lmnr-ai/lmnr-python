# Fix for Issue Tracking When Association Properties is None

## Problem
When `association_properties` parameter is `None` in the `observe_base` and `async_observe_base` decorators, the code attempts to call `.get()` method on None, resulting in an AttributeError: "'NoneType' object has no attribute 'get'".

## Root Cause
In `/src/lmnr/opentelemetry_lib/decorators/__init__.py`, lines 190, 194, 264, and 268 were calling `association_properties.get()` without first checking if `association_properties` is None.

## Solution
Added None checks before attempting to access `association_properties`:

```python
# Before:
if session_id := association_properties.get("session_id"):
    # ...

# After:
if association_properties and (session_id := association_properties.get("session_id")):
    # ...
```

## Changes Made
1. Modified 4 lines in `/src/lmnr/opentelemetry_lib/decorators/__init__.py`:
   - Line 190: Added `association_properties and` check for session_id
   - Line 194: Added `association_properties and` check for user_id
   - Line 264: Added `association_properties and` check for session_id (async)
   - Line 268: Added `association_properties and` check for user_id (async)

2. Added comprehensive test coverage in `tests/test_association_properties_none.py` covering:
   - `observe_base` with None association_properties
   - `async_observe_base` with None association_properties
   - Empty dict association_properties
   - Valid association_properties with values
   - Behavior when Laminar is not initialized

## Testing
- All new tests pass (5/5)
- All existing observe tests pass (39/39)
- No regressions introduced

## Impact
This fix ensures that the observe decorators work correctly when:
1. `association_properties` is explicitly set to None
2. Laminar is initialized but no association properties are provided
3. Using the low-level `observe_base` and `async_observe_base` decorators directly

The fix maintains backward compatibility while preventing the AttributeError.