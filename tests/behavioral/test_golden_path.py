import pytest

from aivalidatorservice.service.service import ModerationService


@pytest.mark.asyncio
async def test_high_logit_returns_true(
    toxic_service: ModerationService,
) -> None:
    result = await toxic_service.moderate("arbitrary text")
    assert result is True


@pytest.mark.asyncio
async def test_low_logit_returns_false(
    clean_service: ModerationService,
) -> None:
    result = await clean_service.moderate("arbitrary text")
    assert result is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "",
        "Привет こんにちは مرحبا",
        "word " * 300,
    ],
)
async def test_crash_safety_inputs_return_boolean(
    clean_service: ModerationService, text: str
) -> None:
    result = await clean_service.moderate(text)
    assert isinstance(result, bool)
