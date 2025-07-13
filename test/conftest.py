import pytest

# Will collect one dict per QA test
_results = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    After each test call, gather any user_properties
    (question, reference, generated, correct, explanation)
    and stash them in _results.
    """
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        props = dict(item.user_properties)
        if props:
            _results.append(props)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    At the end of the run, print a detailed report
    for every question.
    """
    terminalreporter.write_sep("=", "📋 RAG QA Detailed Report")
    for r in _results:
        status = "✅ PASS" if r.get("correct") else "❌ FAIL"
        terminalreporter.write_line(f"{status}: {r.get('question')}")
        terminalreporter.write_line(f"  Reference   : {r.get('reference')}")
        terminalreporter.write_line(f"  Generated   : {r.get('generated')}")
        terminalreporter.write_line(f"  Explanation : {r.get('explanation')}\n")
