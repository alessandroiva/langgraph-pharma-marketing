"""
Error handling utilities with retry logic and circuit breaker pattern.
Demonstrates production-ready error handling and resilience patterns.
"""

import time
import logging
import logging
from typing import Callable, TypeVar, Any, Union, Optional
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


# Custom Exception Hierarchy
class PharmaAIException(Exception):
    """Base exception for all application errors."""
    pass


class LLMException(PharmaAIException):
    """Raised when LLM API calls fail."""
    pass


class RetrievalException(PharmaAIException):
    """Raised when document retrieval fails."""
    pass


class ComplianceException(PharmaAIException):
    """Raised when compliance validation fails."""
    pass


class ValidationException(PharmaAIException):
    """Raised when input validation fails."""
    pass


class ConfigurationException(PharmaAIException):
    """Raised when configuration is invalid."""
    pass


# Retry Decorator with Exponential Backoff
def retry_with_exponential_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: tuple = (LLMException, RetrievalException)
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Demonstrates production-ready retry logic for external API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Circuit Breaker Pattern
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by temporarily disabling calls to failing services.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    
    Demonstrates production resilience patterns.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to track for failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None

        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise PharmaAIException(
                    f"Circuit breaker is OPEN. Service unavailable. "
                    f"Retry after {self.recovery_timeout}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def _on_failure(self) -> None:
        """Handle failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] = Exception
) -> Callable:
    """
    Circuit breaker decorator.
    
    Args:
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        expected_exception: Exception type to track
        
    Returns:
        Decorated function with circuit breaker
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Graceful Degradation
def with_fallback(fallback_value: Any = None, log_error: bool = True) -> Callable:
    """
    Decorator to provide graceful degradation with fallback value.
    
    Args:
        fallback_value: Value to return on error
        log_error: Whether to log the error
        
    Returns:
        Decorated function with fallback behavior
    """
    def decorator(func: Callable[P, T]) -> Callable[P, Union[T, Any]]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[T, Any]:

            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}, using fallback")
                return fallback_value
        return wrapper
    return decorator
