"""Tests for dp5.logger.setup_logger"""

import logging
import io
import pytest

from dp5.logger import setup_logger


class TestSetupLogger:
    def test_returns_logger(self):
        logger = setup_logger(name="test_returns_logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = setup_logger(name="test_named_logger")
        assert logger.name == "test_named_logger"

    def test_default_level_is_info(self):
        logger = setup_logger(name="test_default_level")
        assert logger.level == logging.INFO

    def test_custom_level(self):
        logger = setup_logger(name="test_custom_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_stream_handler_added(self):
        stream = io.StringIO()
        logger = setup_logger(name="test_stream_handler", stream=stream)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_log_message_goes_to_stream(self):
        stream = io.StringIO()
        logger = setup_logger(name="test_log_message", stream=stream, level=logging.DEBUG)
        logger.debug("hello_test_message")
        output = stream.getvalue()
        assert "hello_test_message" in output

    def test_file_handler_created(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = setup_logger(name="test_file_handler", filename=log_file)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        logger.info("file_test_message")
        # Flush and close to ensure data is written
        for h in logger.handlers:
            h.flush()
            h.close()
        with open(log_file) as f:
            content = f.read()
        assert "file_test_message" in content

    def test_propagate_flag(self):
        logger = setup_logger(name="test_propagate_true", propagate=True)
        assert logger.propagate is True

        logger2 = setup_logger(name="test_propagate_false", propagate=False)
        assert logger2.propagate is False

    def test_existing_handlers_cleared(self):
        name = "test_clear_handlers"
        # Set up twice — handlers must not accumulate
        setup_logger(name=name)
        setup_logger(name=name)
        logger = logging.getLogger(name)
        assert len(logger.handlers) == 1

    def test_root_logger_when_name_is_none(self):
        logger = setup_logger(name=None)
        assert logger.name == "root"

    def test_debug_format_flag(self):
        stream = io.StringIO()
        logger = setup_logger(name="test_debug_format", stream=stream, debug=True)
        assert logger is not None

    def test_dict_config(self):
        config = {
            "version": 1,
            "handlers": {},
            "loggers": {},
            "disable_existing_loggers": False,
        }
        # Should not raise
        result = setup_logger(name="test_dict_config", config=config)
        # returns None when using dictConfig branch
        assert result is None
