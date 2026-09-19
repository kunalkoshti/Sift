import logging

from log_api.app import configure_application_logging


def test_configure_application_logging_sets_log_api_level():
    configure_application_logging("INFO")

    application_logger = logging.getLogger("log_api")
    assert application_logger.level == logging.INFO
    assert application_logger.propagate is False
    assert application_logger.handlers
