# conftest.py
import pytest
import time


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # This is a hook that is called before and after each test item
    # We use 'tryfirst=True' to ensure our hook runs before others,
    # and 'hookwrapper=True' to allow us to wrap the execution.

    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        # 'call' refers to the actual execution phase of the test function
        duration_ms = report.duration * 1000  # Convert to milliseconds for clarity

        # Format the duration for printing
        if duration_ms < 1000:
            duration_str = f"{duration_ms:.2f} ms"
        else:
            duration_str = f"{report.duration:.2f} s"

        # Append the duration to the test outcome string
        if report.passed:
            report.nodeid = f"{report.nodeid} PASSED [{duration_str}]"
        elif report.failed:
            report.nodeid = f"{report.nodeid} FAILED [{duration_str}]"
        elif report.skipped:
            report.nodeid = f"{report.nodeid} SKIPPED [{duration_str}]"
        elif report.error:
            report.nodeid = f"{report.nodeid} ERROR [{duration_str}]"
